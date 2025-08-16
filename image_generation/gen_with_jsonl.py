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
    payload = {
        "test_case_id": case["test_case_id"],
        "prompt": case["prompt"],
        "negative_prompt": case.get("negative_prompt", ""),
        "aspect_ratio": "16:9",  # 或者从 case 里推导
        "num_inference_steps": 10,
        "true_cfg_scale": 4.0,
        "seed": case.get("seed", random.randint(0, 100)),
        "language": "en"
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
            img_url = f"{SERVER_URL}{data['result_url']}"
            img_data = requests.get(img_url)
            img_path = os.path.join(OUT_DIR, f"{task_id}.png")
            with open(img_path, "wb") as f:
                f.write(img_data.content)
            print(f"✅ Saved {img_path}")
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
            wait_for_result(tid)