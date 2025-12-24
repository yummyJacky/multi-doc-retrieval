from __future__ import annotations
import os
os.environ['QWEN_VL_SERVER_URL'] = 'http://127.0.0.1:8001'
import base64
import csv
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import requests
from tqdm.auto import tqdm


BASE_DIR = Path(__file__).resolve().parent
ESG_DIR = BASE_DIR / "results" / "dots_ocr" / "2024_Tencent_ESG"
CSV_PATH = ESG_DIR / "2024_Tencent_ESG_page_table_image_summary.csv"
OUTPUT_JSONL = ESG_DIR / "2024_Tencent_ESG_table_qa_qwen2_5vl.jsonl"

logger = logging.getLogger("VisDoMRAG")

# Qwen2.5-VL-7B-Instruct 模型名称（默认），实际服务模型名可通过环境变量覆盖
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_NAME = os.getenv("QWEN_VL_MODEL", DEFAULT_MODEL_NAME)
QWEN_VL_SERVER_URL = os.getenv("QWEN_VL_SERVER_URL")
QWEN_VL_API_KEY = os.getenv("QWEN_VL_API_KEY")


@dataclass
class TablePage:
    filename: str
    page_number: str


def load_table_pages(csv_path: Path) -> List[TablePage]:
    """从 CSV 中读取包含表格的页。

    约定 CSV 至少有字段: filename, page_number, has_table
    has_table 为 "1" 时认为该页包含表格。
    """

    pages: List[TablePage] = []

    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            has_table = str(row.get("has_table", "")).strip().lower()
            if has_table not in {"1", "true", "yes"}:
                continue

            filename = str(row.get("filename", "")).strip()
            page_num = str(row.get("page_number", "")).strip()

            if not page_num:
                m = re.search(r"page_(\d+)", filename)
                if m:
                    page_num = m.group(1)

            if not filename or not page_num:
                continue

            pages.append(TablePage(filename=filename, page_number=page_num))

    return pages


def find_page_images(page_number: str, image_dir: Path) -> List[Path]:
    """根据页码在目录下查找对应的 jpg 页面图片。

    假设整页图片命名类似 ``2024_Tencent_ESG_page_110.jpg``，则：

    - pattern 使用 ``*page_{page_number}.jpg``，这样：
      - page_number=1 时只会匹配 ``...page_1.jpg``，不会匹配 ``...page_10.jpg``；
      - page_number=110 时匹配 ``...page_110.jpg``。
    """

    pattern = f"*page_{page_number}.jpg"

    results: List[Path] = []
    seen = set()

    for p in image_dir.glob(pattern):
        if p not in seen:
            seen.add(p)
            results.append(p)

    return sorted(results)


def build_generation_prompt() -> str:
    """构造给 Qwen2.5-VL 的中文指令，让模型基于表格图片生成 QA 对。"""

    prompt = (
        "你是一名专业的 ESG 报告分析助手。现在给你一张 ESG 报告的页面截图，"
        "其中包含一个或多个表格。请仔细阅读图像中的表格，只根据表格中的数据"
        "来设计 3-5 个有代表性的中文问答对。\n\n"
        "要求：\n"
        "1. 问题和答案必须严格基于表格中的数值或文字信息，不要加入常识推理或主观评价。\n"
        "2. 尽量覆盖表格中的不同维度，例如年份变化、指标类别、数值对比、占比等。\n"
        "3. 每个问题都要清晰具体，答案要简洁准确，直接给出表格中的结论或数据。\n"
        "4. 输出格式必须是 JSON 数组，每个元素形如：{\"question\": \"问题\", \"answer\": \"答案\"}。\n"
        "5. 只输出 JSON，不要输出任何额外说明或文字。"
    )
    return prompt


def parse_qa_response(text: str) -> List[Dict[str, str]]:
    """解析 Qwen 返回的文本为 QA 列表。

    优先假设模型严格输出 JSON 数组；如果解析失败，则退化为单一 QA。"""

    text = (text or "").strip()
    if not text:
        return []

    # 如果模型用 ```json ``` 包裹，先抽取代码块内容
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        candidate = fence_match.group(1).strip()
    else:
        candidate = text

    # 尝试直接解析 JSON
    try:
        data = json.loads(candidate)
        if isinstance(data, dict):
            data = [data]
        result: List[Dict[str, str]] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            q = str(item.get("question", "")).strip()
            a = str(item.get("answer", "")).strip()
            if q and a:
                result.append({"question": q, "answer": a})
        if result:
            return result
    except Exception:
        pass

    # 退化：把整个输出当作一个答案，问题用通用描述
    return [
        {
            "question": "请根据该 ESG 报告表格内容给出一个概括性回答。",
            "answer": text,
        }
    ]


def _normalize_server_url(server_url: str | None) -> str | None:
    """标准化服务 URL，确保包含协议并去掉尾部斜杠。"""

    if not server_url:
        return None
    s = server_url.strip()
    if not (s.startswith("http://") or s.startswith("https://")):
        s = "http://" + s
    return s.rstrip("/")


def _chat_completions_url() -> str | None:
    """返回 /v1/chat/completions 完整 URL。"""

    base = _normalize_server_url(QWEN_VL_SERVER_URL)
    if not base:
        return None
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return f"{base}/chat/completions"


def generate_qa_for_image(image_path: str, prompt: str, max_tokens: int = 512) -> str:
    """调用 Qwen-VL 服务，对单张图片生成表格 QA JSON 文本。"""

    url = _chat_completions_url()
    if url is None:
        logger.warning(
            "QWEN_VL_SERVER_URL not set; cannot call Qwen-VL service for QA generation."
        )
        return ""

    if not os.path.exists(image_path):
        logger.warning("Image file for QA generation not found: %s", image_path)
        return ""

    try:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        logger.error("Error reading image for QA generation %s: %s", image_path, str(e))
        return ""

    # 使用 base64 data URL 通过 image_url 传图像
    data_url = f"data:image/png;base64,{b64}"

    payload: Dict[str, object] = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
    }

    headers = {"Content-Type": "application/json"}
    if QWEN_VL_API_KEY:
        headers["Authorization"] = f"Bearer {QWEN_VL_API_KEY}"

    try:
        response = requests.post(
            url,
            headers=headers,
            data=json.dumps(payload, ensure_ascii=False),
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        logger.error("Error calling Qwen-VL server for %s: %s", image_path, str(e))
        return ""

    try:
        choices = data.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message", {})
        content = message.get("content")

        # vLLM OpenAI server 可能返回字符串或 block 列表
        if isinstance(content, str):
            generated = content
        else:
            parts: List[str] = []
            for block in content or []:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
            generated = "".join(parts)

        return (generated or "").strip()
    except Exception as e:
        logger.error(
            "Error parsing Qwen-VL response for QA generation %s: %s", image_path, str(e)
        )
        return ""


def generate_table_qa() -> Path:
    """主流程：

    1. 读取 CSV 中标记为 has_table=1 的页；
    2. 查找每一页对应的图片文件；
    3. 使用 Qwen2.5-VL-7B-Instruct 生成表格理解类 QA 对；
    4. 将 QA 以 JSONL 格式写入 OUTPUT_JSONL。
    """

    if not ESG_DIR.is_dir():
        raise FileNotFoundError(f"ESG image directory not found: {ESG_DIR}")

    table_pages = load_table_pages(CSV_PATH)
    if not table_pages:
        raise RuntimeError(f"No table pages found in CSV: {CSV_PATH}")

    print(f"Loaded {len(table_pages)} table pages from {CSV_PATH}")

    # 检查服务端配置
    if _chat_completions_url() is None:
        raise RuntimeError(
            "QWEN_VL_SERVER_URL is not set; please configure the Qwen-VL HTTP server first."
        )

    prompt = build_generation_prompt()

    # 输出 JSONL：一行一个 QA 样本
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_JSONL.open("w", encoding="utf-8") as fout:
        for page in tqdm(table_pages, desc="Generating QA for table pages"):
            images = find_page_images(page.page_number, ESG_DIR)
            if not images:
                print(
                    f"[WARN] No images found for page {page.page_number} "
                    f"(filename={page.filename}) in {ESG_DIR}"
                )
                continue

            for img_path in images:
                try:
                    response_text = generate_qa_for_image(str(img_path), prompt)
                except Exception as e:
                    print(f"[ERROR] VQA generation failed for {img_path}: {e}")
                    continue

                qa_items = parse_qa_response(response_text)
                if not qa_items:
                    continue

                for idx, qa in enumerate(qa_items):
                    record = {
                        "doc": "2024_Tencent_ESG",
                        "page_number": page.page_number,
                        "page_filename": page.filename,
                        "image_path": str(img_path.relative_to(ESG_DIR)),
                        "qa_index": idx,
                        "question": qa["question"],
                        "answer": qa["answer"],
                        "has_table": True,
                    }
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"QA dataset written to: {OUTPUT_JSONL}")
    return OUTPUT_JSONL


if __name__ == "__main__":  # pragma: no cover
    generate_table_qa()  # 直接运行脚本时生成 QA 数据集
