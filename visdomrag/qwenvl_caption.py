import os
import re
import logging
from typing import Optional, Dict, Any


class QwenVLCaptioner:
    """Helper for generating image captions using a Qwen-VL model served via vLLM.

    The model checkpoint path should be provided via `checkpoint_path`.
    Captioning is performed lazily: the underlying model is only initialized
    when the first caption request is made.
    """

    def __init__(self, checkpoint_path: Optional[str], logger: logging.Logger) -> None:
        self.checkpoint_path = checkpoint_path
        self.logger = logger

        self._processor = None
        self._process_vision_info = None
        self._llm = None
        self._sampling_params = None

    def _ensure_initialized(self) -> None:
        """Lazily initialize processor, vision helper and vLLM engine."""
        if self._llm is not None:
            return

        if not self.checkpoint_path:
            self.logger.warning(
                "qwen_vl_checkpoint not set in config; image captions will be skipped."
            )
            return

        try:
            from transformers import AutoProcessor
            from qwen_vl_utils import process_vision_info as qwen_vl_process_vision_info
            from vllm import LLM, SamplingParams

            self._processor = AutoProcessor.from_pretrained(self.checkpoint_path)
            self._process_vision_info = qwen_vl_process_vision_info

            self._llm = LLM(
                model=self.checkpoint_path,
                trust_remote_code=True,
                gpu_memory_utilization=0.5,
                enforce_eager=False,
                seed=0,
                max_model_len=32768,
            )

            self._sampling_params = SamplingParams(
                temperature=0,
                max_tokens=256,
                top_k=-1,
                stop_token_ids=[],
            )
            self.logger.info(
                "Initialized Qwen-VL captioner from checkpoint: %s",
                self.checkpoint_path,
            )
        except ImportError as e:
            self.logger.error(
                "Failed to import dependencies for Qwen-VL captioner: %s", str(e)
            )
        except Exception as e:
            self.logger.error("Error initializing Qwen-VL captioner: %s", str(e))

    def _prepare_inputs(self, messages: Any) -> Optional[Dict[str, Any]]:
        """Prepare multimodal inputs for vLLM-based Qwen-VL, following the test notebook."""
        if self._processor is None or self._process_vision_info is None:
            return None

        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs, video_kwargs = self._process_vision_info(
            messages,
            image_patch_size=self._processor.image_processor.patch_size,
            return_video_kwargs=True,
            return_video_metadata=True,
        )

        mm_data: Dict[str, Any] = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs

        return {
            "prompt": text,
            "multi_modal_data": mm_data,
            "mm_processor_kwargs": video_kwargs,
        }

    def caption_image(self, image_path: str) -> str:
        """Generate a short caption for a single image file.

        The prompt follows `test_qwenvl.ipynb`: if the image contains any text or
        tabular data, generate a brief description of the key information;
        otherwise, return an empty caption (treating "无描述" as empty).
        """
        self._ensure_initialized()
        if self._llm is None:
            return ""

        if not os.path.exists(image_path):
            self.logger.warning("Image file for captioning not found: %s", image_path)
            return ""

        prompt = (
            "图片如果包含任何文本或表格数据，生成简短的描述，概述其关键信息；"
            "如果没有，请返回 '无描述'。"
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self._prepare_inputs(messages)
        if inputs is None:
            return ""

        try:
            outputs = self._llm.generate([inputs], sampling_params=self._sampling_params)
            if not outputs:
                return ""

            generated = outputs[0].outputs[0].text if outputs[0].outputs else ""
            caption = (generated or "").strip().strip("'\"")
            if caption == "无描述":
                return ""
            return caption
        except Exception as e:
            self.logger.error(
                "Error generating caption with Qwen-VL for %s: %s", image_path, str(e)
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
