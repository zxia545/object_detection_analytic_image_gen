from datasets import load_dataset
from datasets.features import Image
import os, shutil

# 1) load dataset normally
ds = load_dataset(
    "zxia545/object_detection_analytic_image",
    split="train",
    cache_dir="./huggingface_cache",
)
print(ds)

# 2) re-cast image column to get 'path'
ds = ds.cast_column("image", Image(decode=False))

# 3) output dir
out_dir = "./downloaded_images"
os.makedirs(out_dir, exist_ok=True)

saved, skipped = 0, 0
for row in ds:
    info = row["image"]   # dict: {"path": "...", "bytes": None}
    src_path = info["path"]
    fname = os.path.basename(src_path)
    dst_path = os.path.join(out_dir, fname)

    if os.path.exists(dst_path):
        skipped += 1
        continue

    shutil.copy2(src_path, dst_path)
    saved += 1

print(f"✅ Done. Saved {saved}, Skipped {skipped}")
print(f"Files are in: {os.path.abspath(out_dir)}")
