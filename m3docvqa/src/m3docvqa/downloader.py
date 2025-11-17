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

"""
Downloader Module for M3DocVQA

This module provides functions to download Wikipedia pages in either PDF or PNG format
for the M3DocVQA dataset. It uses Playwright to load and capture the pages in a headless
browser environment and saves each page in the specified format.

Functions:
    - _download_wiki_page: Downloads a single Wikipedia page as a PDF or PNG.
    - download_wiki_page: Manages the downloading of multiple Wikipedia pages.
"""

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
from loguru import logger
from pathlib import Path
import jsonlines
from tqdm.auto import tqdm
from m3docvqa.pdf_utils import is_pdf_downloaded
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, RetryCallState


def log_retry_attempt(retry_state: RetryCallState):
    """回调函数，用于记录每次重试的信息。"""
    logger.warning(
        f"页面下载失败，正在进行第 {retry_state.attempt_number} 次重试... "
        f"(等待 {retry_state.idle_for:.2f} 秒)"
    )


@retry(
    stop=stop_after_attempt(5),  # 最多重试5次（5xx错误需要更多重试）
    wait=wait_exponential(multiplier=2, min=4, max=60),  # 更长的等待时间：4, 8, 16, 32, 60秒
    retry=retry_if_exception_type((PlaywrightTimeoutError, Exception)),  # 重试所有异常
    before_sleep=log_retry_attempt,
    reraise=True
)
def _download_wiki_page(args: tuple[int, int, str, str, str, int]) -> tuple[bool, Exception | None]:
    """Download a single Wikipedia page as a PDF or PNG using Playwright.

    Args:
        args (Tuple[int, int, str, str, str, int]): Contains order in batch, total count, URL, save path,
            save type ('pdf' or 'png'), and process ID.

    Returns:
        Tuple[bool, Optional[Exception]]: A tuple where the first element is a boolean indicating success,
            and the second element is an exception if an error occurred, or None otherwise.
    """
    order_i, total, url, save_path, save_type, proc_id = args
    save_path = Path(save_path)
    
    # 检查文件是否已存在且有效
    if save_path.exists():
        if save_type == 'pdf':
            # 对于PDF，检查文件是否有效
            if is_pdf_downloaded(str(save_path)):
                logger.info(f"File {save_path.name} already exists and is valid, skipping...")
                return True, None
            else:
                logger.warning(f"File {save_path.name} exists but is invalid, re-downloading...")
                save_path.unlink()
        elif save_type == 'png':
            # 对于PNG，检查文件大小是否合理（>1KB）
            if save_path.stat().st_size > 1024:
                logger.info(f"File {save_path.name} already exists, skipping...")
                return True, None
            else:
                logger.warning(f"File {save_path.name} exists but is too small, re-downloading...")
                save_path.unlink()

    # 不使用 try-except，让异常向上传播以触发重试机制
    with sync_playwright() as p:
        # 配置浏览器启动参数（代理在 context 级别设置）
        browser = p.chromium.launch(headless=True)
        
        # 配置 context，包括代理设置
        context_options = {
            'ignore_https_errors': True,
            'user_agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        context = browser.new_context(**context_options)
        page = context.new_page()
        page.set_default_timeout(60000)  # 增加到60秒超时

        # 尝试访问页面，如果遇到5xx错误会抛出异常触发重试
        response = page.goto(url, wait_until='domcontentloaded')  # 先等待DOM加载
        
        # 检查响应状态
        if response and response.status >= 500:
            logger.warning(f"Server error {response.status} for {url}, will retry...")
            browser.close()
            raise Exception(f"Server returned {response.status} error")
        
        # 等待页面完全加载
        try:
            page.wait_for_load_state('networkidle', timeout=30000)
        except:
            # 如果networkidle超时，至少等待页面可交互
            logger.warning(f"Network idle timeout for {url}, continuing anyway...")
            page.wait_for_load_state('domcontentloaded')
        
        if save_type == 'png':
            page.screenshot(path=str(save_path), full_page=True)
        elif save_type == 'pdf':
            page.emulate_media(media="screen")
            page.pdf(path=str(save_path))

        browser.close()

    return True, None


def download_wiki_page(
    urls: list[str],
    save_paths: list[str],
    save_type: str,
    result_log_dir: str,
    proc_id: int = 0,
    n_proc: int = 1,
) -> list[bool]:
    """Download multiple Wikipedia pages and log progress.

    Args:
        urls (List[str]): List of Wikipedia URLs to download.
        save_paths (List[str]): List of paths where each downloaded file will be saved.
        save_type (str): File type to save each page as ('pdf' or 'png').
        result_log_dir (str): Path to the directory where the download results will be logged.
        proc_id (int, optional): Process ID for parallel processing. Defaults to 0.
        n_proc (int, optional): Total number of processes running in parallel. Defaults to 1.

    Returns:
        List[bool]: A list of booleans indicating whether each download was successful.
    """
    total = len(urls)
    all_args = [(i, total, url, str(save_path), save_type, proc_id) 
                for i, (url, save_path) in enumerate(zip(urls, save_paths))]

    # create log directory if it doesn't exist
    log_dir = Path(result_log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(total=len(all_args), ncols=100, disable=not (proc_id == 0))

    results = []
    n_downloaded = 0

    for args in all_args:
        try:
            result = _download_wiki_page(args)
            # _download_wiki_page returns (True, None) on success
            downloaded = result[0] if isinstance(result, tuple) else result
            error = None
            n_downloaded += 1
        except Exception as e:
            downloaded, error = False, e
            logger.error(f"Failed to download after all retries. Error: {e}")

        pbar.set_description(f"Process: {proc_id}/{n_proc} - Downloaded: {n_downloaded}/{total}")
        pbar.update(1)

        results.append(downloaded)
        
        # Write to process-specific log file
        proc_result_path = log_dir / f'process_{proc_id}_{n_proc}.jsonl'
        with jsonlines.open(proc_result_path, mode='a') as writer:  
            writer.write({
                'downloaded': downloaded,
                'args': [arg if not isinstance(arg, Path) else str(arg) for arg in args],
                'error': str(error) if error else None
            })

    pbar.close()
    return results
