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

        pattern = r"!\[]\(([^)]+)\)"
        dir_name = os.path.dirname(md_path)

        def _replace(match: re.Match) -> str:
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
            new_md_text = re.sub(pattern, _replace, md_text)
        except Exception as e:
            self.logger.error(
                "Error applying caption regex to markdown %s: %s", md_path, str(e)
            )
            return md_text

        if new_md_text != md_text:
            try:
                with open(md_path, "w", encoding="utf-8") as f:
                    f.write(new_md_text)
            except Exception as e:
                self.logger.error(
                    "Error writing augmented markdown to %s: %s", md_path, str(e)
                )
        return new_md_text
