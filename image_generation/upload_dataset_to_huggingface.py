from huggingface_hub import HfApi
api = HfApi()

api.upload_folder(
    folder_path="image_generation/gen_images_v1",
    path_in_repo="gen_images_v1",
    repo_id="zxia545/object_detection_analytic_image",
    repo_type="dataset",
    ignore_patterns="**/logs/*.txt", # Ignore all text logs
)