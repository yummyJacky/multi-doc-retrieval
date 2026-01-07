import base64
import json
import logging
import os
import re
from typing import Optional, Dict, Any

import requests


class QwenVLCaptioner:
    """Helper for generating image captions using a Qwen-VL model via a vLLM HTTP server.

    Instead of loading the model locally, this class talks to an external vLLM
    server that exposes an OpenAI-compatible `/v1/chat/completions` endpoint.

    Typical config (in VisDoMRAG):
      - qwen_vl_server_url: base URL of the vLLM server, e.g. "http://127.0.0.1:8000" or "http://127.0.0.1:8000/v1"
      - qwen_vl_model: model name served by vLLM, e.g. "Qwen/Qwen3-VL-4B-Instruct"
      - qwen_vl_api_key: optional API key used by the vLLM OpenAI server
    """

    def __init__(
        self,
        server_url: Optional[str],
        model_name: str,
        logger: logging.Logger,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
    ) -> None:
        # Normalize server URL and ensure it has a scheme so that requests
        # can create a proper adapter (avoid "No connection adapters" errors
        # when users pass values like "127.0.0.1:8001").
        if server_url:
            s = server_url.strip()
            if not (s.startswith("http://") or s.startswith("https://")):
                s = "http://" + s
            self.server_url = s.rstrip("/")
        else:
            self.server_url = None
        self.model_name = model_name
        self.logger = logger
        self.api_key = api_key or os.getenv("QWEN_VL_API_KEY")
        self.timeout = timeout

        # Internal flag to avoid repeating warnings when server URL is missing.
        self._unavailable_warned = False

    def _chat_completions_url(self) -> Optional[str]:
        """Return full URL to the /v1/chat/completions endpoint, or None if unset."""
        if not self.server_url:
            return None

        base = self.server_url.rstrip("/")
        # Normalize so that we always end with /v1
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        return f"{base}/chat/completions"

    def _ensure_server_available(self) -> bool:
        """Check that server URL is configured; log a warning once if not."""
        if self._chat_completions_url() is None:
            if not self._unavailable_warned:
                self.logger.warning(
                    "qwen_vl_server_url not set in config; image captions will be skipped."
                )
                self._unavailable_warned = True
            return False
        return True

    def _classify_image_type(self, image_path: str) -> str:
        """Classify image into logo / chart / photo using Qwen-VL.

        Returns one of "logo", "chart", "photo". If classification fails, defaults to
        "photo" so that we conservatively skip caption generation.
        """
        if not self._ensure_server_available():
            return "photo"

        if not os.path.exists(image_path):
            self.logger.warning(
                "Image file for classification not found: %s", image_path
            )
            return "photo"

        url = self._chat_completions_url()
        if url is None:
            return "photo"

        classify_prompt = (
            "你是一个图像类型分类器。请根据图片内容，在下面三类中选择最合适的一类，"
            "并且最终只输出一个英文单词，不要输出任何其他内容：\n"
            "- logo：公司标志、品牌 Logo、商标、图标或简单装饰图案。\n"
            "- chart：数据图表或曲线图，例如折线图、柱状图、饼图、散点图等，用来展示数值或统计信息。\n"
            "- photo：普通照片或插画，包括人物、风景、城市、产品、示意图等。\n\n"
            "回答时，只能输出下面三个单词之一：logo、chart、photo。"
        )

        try:
            with open(image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            self.logger.error(
                "Error reading image for classification %s: %s", image_path, str(e)
            )
            return "photo"

        data_url = f"data:image/png;base64,{b64}"

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": classify_prompt},
                    ],
                }
            ],
            "max_tokens": 8,
            "temperature": 0,
        }

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            response = requests.post(
                url,
                headers=headers,
                data=json.dumps(payload),
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            self.logger.error(
                "Error calling Qwen-VL classifier for %s: %s", image_path, str(e)
            )
            return "photo"

        try:
            choices = data.get("choices") or []
            if not choices:
                return "photo"
            message = choices[0].get("message", {})
            content = message.get("content")

            if isinstance(content, str):
                generated = content
            else:
                parts = []
                for block in content or []:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block.get("text", ""))
                generated = "".join(parts)

            label = (generated or "").strip().strip("'\"").lower()
            if label not in ("logo", "chart", "photo"):
                return "photo"
            return label
        except Exception as e:
            self.logger.error(
                "Error parsing Qwen-VL classification response for %s: %s",
                image_path,
                str(e),
            )
            return "photo"

    def caption_image(self, image_path: str) -> str:
        """Generate a short caption for a single image file.

        The prompt follows `test_qwenvl.ipynb`: if the image contains any text or
        tabular data, generate a brief description of the key information;
        otherwise, return an empty caption (treating "无描述" as empty).
        """
        if not self._ensure_server_available():
            return ""

        if not os.path.exists(image_path):
            self.logger.warning("Image file for captioning not found: %s", image_path)
            return ""

        # First classify the image type. Only generate captions for chart-like images
        # to avoid wasting time on logos or purely decorative pictures.
        image_type = self._classify_image_type(image_path)
        if image_type != "chart":
            return ""

        url = self._chat_completions_url()
        if url is None:
            return ""

        prompt = (
            "图片如果包含任何文本或表格数据，生成简短的描述，概述其关键信息；如果没有任何文字（包括文本、表格数据等），请必须严格返回 '无描述'。"
        )

        # Encode image as base64 data URL so that the vLLM OpenAI server can
        # receive it as an `image_url` content block.
        try:
            with open(image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            self.logger.error(
                "Error reading image for captioning %s: %s", image_path, str(e)
            )
            return ""

        data_url = f"data:image/png;base64,{b64}"

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": 256,
            "temperature": 0,
        }

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            response = requests.post(
                url,
                headers=headers,
                data=json.dumps(payload),
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            self.logger.error(
                "Error calling Qwen-VL vLLM server for %s: %s", image_path, str(e)
            )
            return ""

        try:
            choices = data.get("choices") or []
            if not choices:
                return ""
            message = choices[0].get("message", {})
            content = message.get("content")

            # vLLM OpenAI server may return either a string or a list of blocks
            if isinstance(content, str):
                generated = content
            else:
                parts = []
                for block in content or []:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block.get("text", ""))
                generated = "".join(parts)

            caption = (generated or "").strip().strip("'\"")
            if caption == "无描述":
                return ""
            return caption
        except Exception as e:
            self.logger.error(
                "Error parsing Qwen-VL response for %s: %s", image_path, str(e)
            )
            return ""

    def caption_table_from_page_image(self, md_path: str, table_index: int) -> str:
        """Generate a table-focused caption from a full-page image inferred from a
        DotsOCR markdown path, for a specific table index on that page.

        This is used when the markdown for a page contains one or more HTML table
        blocks ("<table>...</table>"). We infer the page-level image path from the
        markdown filename (e.g. "XXX_page_10.md" -> "XXX_page_10.png") and ask
        Qwen-VL to describe the key information in the *N-th* table on that page,
        where N is given by ``table_index`` (1-based).
        """
        if not self._ensure_server_available():
            return ""

        if not md_path:
            return ""

        dir_name = os.path.dirname(md_path)
        base_name = os.path.basename(md_path)

        # Support both XXX_page_10.md and XXX_page_10_nohf.md
        m = re.match(r"(.+)_page_(\d+)(?:_nohf)?\.md$", base_name)
        if not m:
            self.logger.debug(
                "Could not infer page image path from markdown path: %s", md_path
            )
            return ""

        pdf_stem, page_idx = m.group(1), m.group(2)
        image_name = f"{pdf_stem}_page_{page_idx}.jpg"
        image_path = os.path.join(dir_name, image_name)

        if not os.path.exists(image_path):
            self.logger.warning(
                "Page image for table caption not found: %s", image_path
            )
            return ""

        url = self._chat_completions_url()
        if url is None:
            return ""

        prompt = (
            f"这是一页 PDF 的整体截图，其中包含一个或多个表格。请重点关注第 {table_index} 个表格，"
            "并用中文精准凝练地概括该表格的核心内容。如果无法清晰看清表格内容，请说明看不清，而不要编造具体信息。"
        )

        try:
            with open(image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            self.logger.error(
                "Error reading page image for table caption %s: %s", image_path, str(e)
            )
            return ""

        data_url = f"data:image/png;base64,{b64}"

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": 512,
            "temperature": 0,
        }

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            response = requests.post(
                url,
                headers=headers,
                data=json.dumps(payload),
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            self.logger.error(
                "Error calling Qwen-VL vLLM server for table caption %s: %s",
                image_path,
                str(e),
            )
            return ""

        try:
            choices = data.get("choices") or []
            if not choices:
                return ""
            message = choices[0].get("message", {})
            content = message.get("content")

            if isinstance(content, str):
                generated = content
            else:
                parts = []
                for block in content or []:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block.get("text", ""))
                generated = "".join(parts)

            caption = (generated or "").strip().strip("'\"")
            return caption
        except Exception as e:
            self.logger.error(
                "Error parsing Qwen-VL response for table caption %s: %s",
                image_path,
                str(e),
            )
            return ""

    def augment_markdown(self, md_path: str) -> str:
        """Append Qwen-VL captions after images in a markdown file.

        For each pattern like `![](relative_image_path.png)`, this will:
        - Resolve the image path relative to the markdown file location.
        - Call Qwen-VL captioner on the image.
        - If a non-empty caption is returned, rewrite the markdown as:

            ![](relative_image_path.png)\ncaption: <caption>

        The modified markdown is written back to `md_path`, and the final text is
        returned. If captioning is unavailable or fails, the original markdown
        content is returned unchanged.
        """
        try:
            with open(md_path, "r", encoding="utf-8") as f:
                md_text = f.read()
        except Exception as e:
            self.logger.error(
                "Error reading markdown for caption augmentation %s: %s", md_path, str(e)
            )
            return ""

        # First, augment image references with captions as before.
        image_pattern = r"!\[]\(([^)]+)\)"
        dir_name = os.path.dirname(md_path)

        def _replace_image(match: re.Match) -> str:
            rel_path = match.group(1)
            full_path = os.path.join(dir_name, rel_path)
            if not os.path.exists(full_path):
                return match.group(0)

            caption = self.caption_image(full_path)
            if not caption:
                return match.group(0)

            # Put caption on its own line after the image reference.
            return f"{match.group(0)}\ncaption: {caption}"

        try:
            processed_md_text = re.sub(image_pattern, _replace_image, md_text)
        except Exception as e:
            self.logger.error(
                "Error applying image caption regex to markdown %s: %s", md_path, str(e)
            )
            processed_md_text = md_text

        # Then, if there are HTML table blocks, generate a table-focused caption from the full-page image and append it after each table. 
        # Each table will get its own caption by calling Qwen-VL separately with a table index.
        table_pattern = re.compile(r"<table\b.*?</table>", re.IGNORECASE | re.DOTALL)

        if table_pattern.search(processed_md_text):
            table_counter = {"idx": 0}

            def _add_table_caption(match: re.Match) -> str:
                table_counter["idx"] += 1
                print("find a table,add table caption...")
                idx = table_counter["idx"]
                table_caption = self.caption_table_from_page_image(md_path, idx)
                table_html = match.group(0)
                if not table_caption:
                    return table_html
                return f"{table_html}\ncaption: {table_caption}"

            try:
                processed_md_text = table_pattern.sub(_add_table_caption, processed_md_text)
            except Exception as e:
                self.logger.error(
                    "Error applying table caption regex to markdown %s: %s",
                    md_path,
                    str(e),
                )

        if processed_md_text != md_text:
            try:
                with open(md_path, "w", encoding="utf-8") as f:
                    f.write(processed_md_text)
            except Exception as e:
                self.logger.error(
                    "Error writing augmented markdown to %s: %s", md_path, str(e)
                )
        return processed_md_text
