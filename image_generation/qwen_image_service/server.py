import os
import uuid
import subprocess
from typing import Optional, Dict, Any

import torch
import asyncio
import requests
import psutil

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from fastapi.responses import FileResponse
import uvicorn

import random
###############################################################################
# 配置参数
###############################################################################
MODEL_PATHS = ["/root/autodl-tmp/Qwen-Image", "/root/autodl-fs/Qwen-Image"]
OUTPUT_DIR = "/root/autodl-tmp/output_images"
MIN_FREE_GB = 5  # 如果可用空间小于10GB则清理

positive_magic = {
    "en": "",
    "zh": " 超清，4K，电影级构图",
}
aspect_ratios = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1140),
    "3:4": (1140, 1472),
}

###############################################################################
# 工具函数
###############################################################################
def make_sure_model():
    for path in MODEL_PATHS:
        if os.path.exists(path):
            print(f"Using model: {path}")
            return path
    # 若均没有则自动下载
    print("Model not found. Running /root/download_model.sh ...")
    code = subprocess.call(['bash', '/root/download_model.sh'])
    # if code != 0:
    #     raise RuntimeError('/root/download_model.sh failed!')
    for path in MODEL_PATHS:
        if os.path.exists(path):
            print(f"Downloaded and using model: {path}")
            return path
    raise RuntimeError('Model still not found after download!')

def check_cuda_offload():  # 大于30G显存不开启offload
    if not torch.cuda.is_available():
        return True
    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    return total_gb < 30

def clean_old_images():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        return
    disk = psutil.disk_usage(OUTPUT_DIR)
    freespace_gb = disk.free / (1024 ** 3)
    if freespace_gb < MIN_FREE_GB:
        files = [(f, os.path.getctime(os.path.join(OUTPUT_DIR, f)), os.path.getsize(os.path.join(OUTPUT_DIR, f))) 
                 for f in os.listdir(OUTPUT_DIR) if os.path.isfile(os.path.join(OUTPUT_DIR, f))]
        files.sort(key=lambda x: (x[2], x[1]), reverse=True)  # 大文件/旧文件优先删
        for f, _, _ in files:
            try:
                os.remove(os.path.join(OUTPUT_DIR, f))
                print(f"Cleanup: deleted {f}")
            except:
                continue
            disk = psutil.disk_usage(OUTPUT_DIR)
            if disk.free / (1024 ** 3) >= MIN_FREE_GB:
                break

###############################################################################
# 全局状态变量
###############################################################################
tasks: Dict[str, Dict[str, Any]] = {}
task_lock = asyncio.Lock()
pipe = None                       # pipeline 对象
model_loaded = False
pipe_lock = asyncio.Lock()        # 推理串行锁

###############################################################################
# Pydantic 数据模型
###############################################################################
class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = ""
    aspect_ratio: Optional[str] = "16:9"
    num_inference_steps: Optional[int] = 35
    true_cfg_scale: Optional[float] = 4.0
    seed: Optional[int] = 42
    language: Optional[str] = "en"
    callback_url: Optional[str] = None

class GenerateStatusResponse(BaseModel):
    status: str
    result_url: Optional[str] = None
    detail: Optional[str] = None
    
class GenerateCaseRequest(BaseModel):
    test_case_id: str
    prompt: str
    negative_prompt: Optional[str] = ""
    aspect_ratio: Optional[str] = "16:9"
    num_inference_steps: Optional[int] = 10
    true_cfg_scale: Optional[float] = 4.0
    seed: Optional[int] = random.randint(0, 100)
    language: Optional[str] = "en"
###############################################################################
# FastAPI/lifespan
###############################################################################
@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = make_sure_model()
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    app.state.model_path = model_path
    yield
    # 可在此释放模型资源

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)
app.mount("/static", StaticFiles(directory="static"), name="static")
###############################################################################
# Pipeline 加载与 inference 调用实现
###############################################################################
def safe_load_pipe(model_path, cpu_offload):
    from diffusers import DiffusionPipeline, QwenImageTransformer2DModel
    from transformers.modeling_utils import no_init_weights
    from dfloat11 import DFloat11Model

    with no_init_weights():
        transformer = QwenImageTransformer2DModel.from_config(
            QwenImageTransformer2DModel.load_config(
                "Qwen/Qwen-Image", subfolder="transformer",local_files_only=True
            ),local_files_only=True
        ).to(torch.bfloat16)
    
    DFloat11Model.from_pretrained(
        "/root/DFloat11__Qwen-Image-DF11",
        device="cpu",
        cpu_offload=cpu_offload,
        pin_memory=False,
        bfloat16_model=transformer,
        local_files_only=True
    )
    
    pipe = DiffusionPipeline.from_pretrained(
        "/root/autodl-tmp/Qwen-Image",
        transformer=transformer,
        torch_dtype=torch.bfloat16,
        # cache_dir="/root/autodl-tmp/Qwen-Image",
        local_files_only=True
    )
    pipe.enable_model_cpu_offload()
    return pipe

def _generate(
    prompt, negative_prompt, aspect_ratio, num_inference_steps, true_cfg_scale, seed, language, output_file, cpu_offload, model_path
):
    from torch import Generator
    global pipe, model_loaded

    # 只加载一次pipeline
    if not model_loaded or pipe is None:
        print("load model!")
        pipe = safe_load_pipe(model_path, cpu_offload)
        model_loaded = True
    width, height = aspect_ratios.get(aspect_ratio, (1664, 928))
    p_prompt = prompt + positive_magic.get(language, positive_magic["en"])
    with torch.inference_mode():
        image = pipe(
            prompt=p_prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            true_cfg_scale=true_cfg_scale,
            generator=Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)
        ).images[0]
        image.save(output_file)

###############################################################################
# API - 1. 生成任务接口 (异步/排队)
###############################################################################

@app.post("/generate_case")
async def generate_case(req: GenerateCaseRequest, background_tasks: BackgroundTasks):
    clean_old_images()
    task_id = req.test_case_id  # 直接用 test_case_id 做任务 ID
    task_info = {
        "status": "pending",
        "result_url": None,
        "detail": None,
        "output_file": f"{OUTPUT_DIR}/{task_id}.png"
    }
    async with task_lock:
        tasks[task_id] = task_info

    async def do_generate():
        try:
            cpu_offload = check_cuda_offload()
            async with pipe_lock:
                async with task_lock:
                    tasks[task_id]["status"] = "running"
                await asyncio.get_event_loop().run_in_executor(
                    None, _generate,
                    req.prompt, req.negative_prompt, req.aspect_ratio, req.num_inference_steps,
                    req.true_cfg_scale, req.seed, req.language,
                    task_info["output_file"], cpu_offload, app.state.model_path
                )
            async with task_lock:
                tasks[task_id]["status"] = "done"
                tasks[task_id]["result_url"] = f"/result/{task_id}.png"
        except Exception as e:
            async with task_lock:
                tasks[task_id]["status"] = "failed"
                tasks[task_id]["detail"] = str(e)

    background_tasks.add_task(do_generate)
    return {"task_id": task_id}

@app.post("/generate")
async def generate_image(req: GenerateRequest, background_tasks: BackgroundTasks):
    clean_old_images()
    task_id = str(uuid.uuid4())
    task_info = {
        "status": "pending",
        "result_url": None,
        "detail": None,
        "callback_url": req.callback_url,
        "output_file": f"{OUTPUT_DIR}/{task_id}.png"
    }
    async with task_lock:
        tasks[task_id] = task_info
    
    async def do_generate():
        async with task_lock:
            tasks[task_id]["status"] = "pending"
        try:
            cpu_offload = check_cuda_offload()
            # pipeline 串行锁!
            async with pipe_lock:
                async with task_lock:
                    tasks[task_id]["status"] = "running"
                await asyncio.get_event_loop().run_in_executor(
                    None, _generate,
                    req.prompt, req.negative_prompt, req.aspect_ratio, req.num_inference_steps,
                    req.true_cfg_scale, req.seed, req.language,
                    task_info["output_file"], cpu_offload, app.state.model_path
                )
            async with task_lock:
                tasks[task_id]["status"] = "done"
                tasks[task_id]["result_url"] = f"/result/{task_id}.png"
            # 回调
            if req.callback_url:
                try:
                    requests.post(req.callback_url, json={"task_id": task_id, "result_url": tasks[task_id]["result_url"]})
                except Exception as e:
                    print(f"Callback failed: {e}")
        except Exception as e:
            async with task_lock:
                tasks[task_id]["status"] = "failed"
                tasks[task_id]["detail"] = str(e)
    background_tasks.add_task(do_generate)
    return {"task_id": task_id}

###############################################################################
# API - 2. 查询任务状态
###############################################################################
@app.get("/status/{task_id}", response_model=GenerateStatusResponse)
async def check_status(task_id: str):
    async with task_lock:
        info = tasks.get(task_id)
        if not info:
            raise HTTPException(status_code=404, detail="Task not found")
        return GenerateStatusResponse(
            status=info['status'],
            result_url=info['result_url'],
            detail=info.get('detail')
        )

###############################################################################
# API - 3. 图片下载
###############################################################################
@app.get("/result/{filename}")
async def get_result(filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path, media_type="image/png")

###############################################################################
# API - 4. 健康检查
###############################################################################
@app.get("/healthz")
async def health_check():
    async with task_lock:
        queue_length = sum(1 for t in tasks.values() if t["status"] in {"pending", "running"})
    return {
        "status": "ok",
        "queue_length": queue_length
    }

@app.get("/queue")
async def get_queue():
    """
    返回当前排队中的任务（pending/running）
    """
    async with task_lock:
        result = [ dict(task_id=k, **v) for k,v in tasks.items() if v['status'] in ('pending', 'running') ]
    return result

@app.get("/history")
async def get_history(skip: int = 0, limit: int = 30):
    """
    返回已完成或失败的历史任务，带分页
    """
    async with task_lock:
        # 按时间排序，最新优先
        result = [ 
            dict(task_id=k, **v) 
            for k,v in tasks.items() if v['status'] in ('done','failed')
        ]
        # 核心修改：计算ctime
        for x in result:
            if os.path.exists(x['output_file']):
                x['ctime'] = os.path.getctime(x['output_file'])
            else:
                x['ctime'] = None
        result.sort(key=lambda x: x['ctime'] or 0, reverse=True)
    return {
        "total": len(result),
        "items": result[skip:skip+limit]
    }

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/static/index.html")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6006)