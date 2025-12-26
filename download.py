import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from huggingface_hub import snapshot_download
from transformers import AutoModel
import argparse


def download_hf_model(model_name: str, local_path: str):
    """
    将 Hugging Face 模型下载到指定的本地目录。
    """
    print(f"--- 准备下载模型: {model_name} ---")
    print(f"--- 目标本地路径: {local_path} ---")
    
    # 获取 Hugging Face Token，如果设置了的话
    # hf_token = os.getenv("HF_TOKEN")

    try:
        # 使用 snapshot_download 下载模型的所有文件
        local_dir = snapshot_download(
            repo_id=model_name,
            cache_dir=local_path, # 指定本地缓存的根目录
            local_dir_use_symlinks=False, # 避免使用符号链接，直接下载文件
            # token=hf_token,
        )
        # model = AutoModel.from_pretrained(
        #     model_name,
        #     cache_dir=local_path,
        #     local_files_only=True,
        # )
        print("\n✅ 模型下载成功！")
        print(f"模型文件实际位于: {local_dir}")
        print("----------------------------------------")

    except Exception as e:
        print(f"\n❌ 模型下载失败: {e}")
        print("请检查网络连接、模型名称或 HF_TOKEN 是否正确。")


if __name__ == "__main__":
    TARGET_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct" 
    
    # 您希望将模型文件存储到的本地指定目录
    TARGET_LOCAL_PATH = "/mnt/SSD2_8TB/zechuan/huggingface" 
    
    # 确保目标目录存在
    os.makedirs(TARGET_LOCAL_PATH, exist_ok=True)
    
    download_hf_model(TARGET_MODEL_NAME, TARGET_LOCAL_PATH)