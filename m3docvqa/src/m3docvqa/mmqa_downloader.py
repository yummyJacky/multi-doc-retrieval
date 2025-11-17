# Copyright 2024 Bloomberg Finance L.P.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Downloads the portion of the multimodalqa dataset from https://github.com/allenai/multimodalqa/tree/master/dataset 
that is useful for creating the m3docvqa dataset.
"""
import gzip
import os
import requests
from loguru import logger
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, RetryCallState 


# --- 新增：重试失败时的日志记录函数 ---
def log_retry_attempt(retry_state: RetryCallState):
    """回调函数，用于记录每次重试的信息。"""
    logger.warning(
        f"下载失败，正在进行第 {retry_state.attempt_number} 次重试... "
        f"(等待 {retry_state.idle_for:.2f} 秒, 异常: {type(retry_state.outcome.exception()).__name__})"
    )

@retry(
    # 停止条件：最多重试 5 次
    stop=stop_after_attempt(5),
    # 等待策略：指数退避，每次等待 1, 2, 4, 8, ... 秒，最大等待 60 秒
    wait=wait_exponential(multiplier=1, min=2, max=60),
    # 重试条件：如果抛出 requests.exceptions.RequestException 或其子类（包括 ConnectionError 和 IncompleteRead）
    retry=retry_if_exception_type(requests.exceptions.RequestException),
    # 每次重试时调用日志函数
    before_sleep=log_retry_attempt,
    # 最终失败时抛出异常
    reraise=True 
)
def download_file(url: str, output_path: str) -> None:
    """Downloads a file from a given URL and saves it to the specified output path (带重试机制和断点续传).

    Args:
        url (str): The URL of the file to download.
        output_path (str): The path where the downloaded file will be saved.

    Raises:
        requests.exceptions.RequestException: If the file could not be downloaded after all retries.
    """
    output_path = Path(output_path)
    temp_path = output_path.with_suffix(output_path.suffix + '.tmp')
    
    # Check if there's a partial download
    resume_byte_pos = 0
    if temp_path.exists():
        resume_byte_pos = temp_path.stat().st_size
        logger.info(f"Found partial download, resuming from byte {resume_byte_pos}")
    
    # Set up headers for resume
    headers = {}
    if resume_byte_pos > 0:
        headers['Range'] = f'bytes={resume_byte_pos}-'
    
    logger.info(f"Attempting to download file from {url} to {output_path}...")
    
    try:
        response = requests.get(url, stream=True, timeout=300, headers=headers)
        response.raise_for_status()
        
        # Get total file size
        total_size = int(response.headers.get('content-length', 0))
        if resume_byte_pos > 0 and response.status_code == 206:
            # Partial content response
            total_size += resume_byte_pos
            logger.info(f"Resuming download: {resume_byte_pos}/{total_size} bytes already downloaded")
        elif resume_byte_pos > 0:
            # Server doesn't support resume, start over
            logger.warning("Server doesn't support resume, starting download from beginning")
            resume_byte_pos = 0
            temp_path.unlink(missing_ok=True)
        
        # Download with progress tracking
        mode = 'ab' if resume_byte_pos > 0 else 'wb'
        downloaded = resume_byte_pos
        
        with open(temp_path, mode) as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    # Log progress every 10MB
                    if downloaded % (10 * 1024 * 1024) < 8192:
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            logger.info(f"Download progress: {downloaded}/{total_size} bytes ({progress:.1f}%)")
                        else:
                            logger.info(f"Downloaded: {downloaded} bytes")
        
        # Move temp file to final location
        temp_path.rename(output_path)
        logger.info(f"File downloaded successfully: {output_path}")
        
    except Exception as e:
        # Keep the temp file for resume on next retry
        logger.error(f"Download interrupted: {e}")
        raise

def decompress_gz_file(input_path: str | Path, output_path: str | Path) -> None:
    """
    Decompresses a `.gz` file into its original format.

    Args:
        input_path (str | Path): Path to the `.gz` file.
        output_path (str | Path): Path where the decompressed file will be written.

    Raises:
        ValueError: If the input path does not exist or is not a file.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.is_file():
        raise ValueError(f"The input file {input_path} does not exist or is not a file.")

    with gzip.open(input_path, "rb") as f_in, open(output_path, "wb") as f_out:
        f_out.write(f_in.read())
    logger.info(f"Decompressed {input_path} to {output_path}")

def download_and_decompress_mmqa(output_directory: str | Path) -> None:
    """
    Downloads and decompresses the MultiModalQA dataset files into the specified directory.

    Args:
        output_directory (str | Path): The directory where the files will be stored.

    Steps:
        1. Creates the output directory if it doesn't exist.
        2. Downloads the `.jsonl.gz` files.
        3. Decompresses each `.gz` file into its `.jsonl` format.
        4. Removes the `.gz` files after decompression.

    Raises:
        requests.exceptions.RequestException: If any of the files could not be downloaded.
    """
    # Define base URL and file names
    base_url = "https://github.com/allenai/multimodalqa/raw/refs/heads/master/dataset/"
    files = [
        "MMQA_texts.jsonl.gz", 
        "MMQA_tables.jsonl.gz", 
        "MMQA_images.jsonl.gz",
        "MMQA_dev.jsonl.gz",
        "MMQA_train.jsonl.gz",
        ]
        
    output_directory = Path(output_directory)

    # Ensure the output directory exists
    if not output_directory.exists():
        output_directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created output directory: {output_directory}")

    for file_name in files:
        compressed_path = output_directory / file_name
        decompressed_path = output_directory / file_name.replace(".gz", "")

        try:
            # Check if already decompressed
            if decompressed_path.exists():
                logger.info(f"File {decompressed_path.name} already exists, skipping...")
                continue
            
            # Step 1: Download the file (skip if already downloaded)
            if not compressed_path.exists():
                logger.info(f"Downloading {file_name}...")
                download_file(base_url + file_name, compressed_path)
            else:
                logger.info(f"Compressed file {file_name} already exists, skipping download...")

            # Step 2: Decompress the file
            logger.info(f"Decompressing {file_name}...")
            decompress_gz_file(compressed_path, decompressed_path)

            # Step 3: Remove the compressed file
            compressed_path.unlink()
            logger.info(f"Removed compressed file: {compressed_path}")
        except Exception as e:
            logger.error(f"Error processing {file_name}: {e}")
            raise
