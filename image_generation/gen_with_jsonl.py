import json
import requests
import time
import os
import random
import argparse

SERVER_URL = "http://localhost:6006"
JSONL_PATH = "od_synth_cases_10000_cctv.jsonl"
OUT_DIR = "gen_images"
os.makedirs(OUT_DIR, exist_ok=True)

def submit_case(case):
    # Check if output image already exists
    task_id = case["test_case_id"]  # Use test_case_id as task_id for consistency
    img_path = os.path.join(OUT_DIR, f"{task_id}.png")
    
    if os.path.exists(img_path):
        print(f"⏭️  Skipping {task_id} - image already exists: {img_path}")
        return None
    
    payload = {
        "test_case_id": case["test_case_id"],
        "prompt": case["prompt"],
        "negative_prompt": case.get("negative_prompt", ""),
        "aspect_ratio": "16:9",  # 或者从 case 里推导
        "num_inference_steps": 30,
        "true_cfg_scale": 4.0,
        "seed": case.get("seed", random.randint(0, 100)),
        "language": "en",
        "output_dir": OUT_DIR  # 新增：指定输出目录
    }
    r = requests.post(f"{SERVER_URL}/generate_case", json=payload)
    r.raise_for_status()
    return r.json()["task_id"]

def wait_for_result(task_id):
    while True:
        r = requests.get(f"{SERVER_URL}/status/{task_id}")
        r.raise_for_status()
        data = r.json()
        if data["status"] == "done":
            # 图片已经直接保存到指定目录，不需要再下载
            img_path = os.path.join(OUT_DIR, f"{task_id}.png")
            if os.path.exists(img_path):
                print(f"✅ Image saved to {img_path}")
            else:
                print(f"⚠️  Image generation completed but file not found at {img_path}")
            break
        elif data["status"] == "failed":
            print(f"❌ Failed {task_id}: {data.get('detail')}")
            break
        else:
            time.sleep(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_path", type=str, default="od_synth_cases_10000_cctv.jsonl")
    parser.add_argument("--out_dir", type=str, default="gen_images")
    args = parser.parse_args()
    JSONL_PATH = args.jsonl_path
    OUT_DIR = args.out_dir
    
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            case = json.loads(line)
            tid = submit_case(case)
            if tid:  # Only wait for result if case was submitted
                wait_for_result(tid)