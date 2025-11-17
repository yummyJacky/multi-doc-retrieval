import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download
# model_path = snapshot_download(repo_id="vidore/colpaligemma-3b-pt-448-base")
target_directory = "./job/model/colpaligemma-3b-pt-448-base"
model_path = snapshot_download(
    repo_id="vidore/colpaligemma-3b-pt-448-base",
    local_dir=target_directory,
    local_dir_use_symlinks=False
    )
