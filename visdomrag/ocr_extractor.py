import os
import tempfile
import shutil
import traceback
from typing import List, Optional

import torch
from pdf2image import convert_from_path
from transformers import AutoModel, AutoTokenizer

from dots_ocr.parser import DotsOCRParser


class OCRExtractor:
    """Helper for extracting per-page text from PDFs using OCR engines.

    This wraps the existing DotsOCR and DeepSeek-OCR logic so that the main
    VisDoMRAG class can delegate OCR responsibilities to this module.
    """

    def __init__(self, config, output_dir: str, logger) -> None:
        self.config = config
        self.output_dir = output_dir
        self.logger = logger

        # DeepSeek-OCR handles (lazily initialized)
        self._deepseek_model: Optional[AutoModel] = None
        self._deepseek_tokenizer: Optional[AutoTokenizer] = None

    # ----------------------- Public API -----------------------
    def extract_text_from_pdf(self, pdf_path: str, ocr_engine: str = "dots", captioner=None) -> List[str]:
        """Extract text from a PDF file using the configured OCR engine.

        Args:
            pdf_path: Path to the PDF file.
            ocr_engine: Either "dots" or "deepseek" (case-insensitive).
            captioner: Optional Qwen-VL captioner object with an augment_markdown(md_path) method.

        Returns:
            List of per-page strings in the "--- Page N ---\n..." format.
        """
        engine = ocr_engine.lower() if isinstance(ocr_engine, str) else "dots"
        if engine == "deepseek":
            return self._extract_text_from_pdf_deepseek(pdf_path)
        else:
            return self._extract_text_from_pdf_dots(pdf_path, captioner=captioner)

    # ----------------------- DotsOCR path -----------------------
    def _extract_text_from_pdf_dots(self, pdf_path: str, captioner=None) -> List[str]:
        try:
            dots_output_dir = os.path.join(self.output_dir, "dots_ocr")
            os.makedirs(dots_output_dir, exist_ok=True)

            dots_kwargs = {
                "max_completion_tokens": self.config.get("dots_max_completion_tokens", 4096),
                "num_thread": self.config.get("dots_num_thread", 16),
                "dpi": self.config.get("dots_dpi", 200),
                "output_dir": dots_output_dir,
            }

            self.logger.info("Using DotsOCR to extract text from PDF: %s", pdf_path)
            dots_ocr_parser = DotsOCRParser(**dots_kwargs)

            prompt_mode = self.config.get("dots_prompt_mode", "prompt_layout_all_en")
            results = dots_ocr_parser.parse_file(
                pdf_path,
                output_dir=dots_output_dir,
                prompt_mode=prompt_mode,
            )

            pages: List[str] = []
            for res in sorted(results, key=lambda r: r.get("page_no", 0)):
                md_path = res.get("md_content_path") or res.get("md_content_nohf_path")
                page_no = res.get("page_no", len(pages)) + 1
                page_text = ""

                if md_path and os.path.exists(md_path):
                    try:
                        if captioner is not None:
                            # Let the captioner read, modify, and return markdown
                            page_text = captioner.augment_markdown(md_path) or ""
                        else:
                            with open(md_path, "r", encoding="utf-8") as f:
                                page_text = f.read()
                    except Exception as read_err:
                        self.logger.error(
                            "Error reading DotsOCR markdown for %s page %d: %s",
                            pdf_path,
                            page_no,
                            read_err,
                        )
                pages.append(f"--- Page {page_no} ---\n{page_text}\n")

            return pages

        except Exception as e:
            self.logger.error(
                "Error extracting text from %s with DotsOCR: %s", pdf_path, str(e)
            )
            traceback.print_exc()
            return []

    # ----------------------- DeepSeek-OCR path -----------------------
    def _initialize_deepseek_ocr(self) -> None:
        """Lazily load DeepSeek-OCR model and tokenizer for OCR-based text extraction."""
        if self._deepseek_model is not None and self._deepseek_tokenizer is not None:
            return

        model_name = self.config.get("deepseek_ocr_model", "deepseek-ai/DeepSeek-OCR")
        self.logger.info("Loading DeepSeek-OCR model: %s", model_name)

        try:
            self._deepseek_tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
            )

            torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_safetensors=True,
                attn_implementation="eager",
                torch_dtype=torch_dtype,
            ).eval()

            if torch.cuda.is_available():
                model = model.to("cuda")

            self._deepseek_model = model
            self.logger.info("DeepSeek-OCR model loaded successfully")
        except Exception as e:
            self.logger.error("Error loading DeepSeek-OCR model %s: %s", model_name, str(e))
            traceback.print_exc()
            raise

    def _extract_text_from_pdf_deepseek(self, pdf_path: str) -> List[str]:
        """Extract per-page text from PDF using DeepSeek-OCR.

        Returns page texts in the same "--- Page N ---" format used by the DotsOCR
        path, but without any image captioning.
        """
        try:
            try:
                self._initialize_deepseek_ocr()
            except Exception:
                # If DeepSeek-OCR fails to initialize, fall back to empty result
                self.logger.warning(
                    "DeepSeek-OCR initialization failed for %s; returning empty result", pdf_path
                )
                return []

            dpi = int(self.config.get("deepseek_dpi", 200))
            base_size = int(self.config.get("deepseek_base_size", 1024))
            image_size = int(self.config.get("deepseek_image_size", 640))
            crop_mode = bool(self.config.get("deepseek_crop_mode", True))

            self.logger.info(
                "Using DeepSeek-OCR to extract text from PDF: %s (dpi=%d)",
                pdf_path,
                dpi,
            )
            pages_img = convert_from_path(pdf_path, dpi=dpi)
            total_pages = len(pages_img)
            if total_pages == 0:
                self.logger.warning(
                    "No pages found when converting PDF %s to images", pdf_path
                )
                return []

            pages: List[str] = []
            prompt_text = "<image>\nFree OCR."

            for page_idx, img in enumerate(pages_img):
                tmp_img = None
                out_dir = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                        img.save(tmp, format="PNG")
                        tmp_img = tmp.name

                    out_dir = tempfile.mkdtemp(prefix="dsocr_pdf_")

                    res = self._deepseek_model.infer(
                        self._deepseek_tokenizer,
                        prompt=prompt_text,
                        image_file=tmp_img,
                        output_path=out_dir,
                        base_size=base_size,
                        image_size=image_size,
                        crop_mode=crop_mode,
                        save_results=False,
                        test_compress=False,
                        eval_mode=True,
                    )

                    if isinstance(res, str):
                        text = res.strip()
                    elif isinstance(res, dict) and "text" in res:
                        text = str(res["text"]).strip()
                    elif isinstance(res, (list, tuple)):
                        text = "\n".join(map(str, res)).strip()
                    else:
                        text = ""

                    if not text:
                        mmd = os.path.join(out_dir, "result.mmd")
                        if os.path.exists(mmd):
                            with open(mmd, "r", encoding="utf-8") as fh:
                                text = fh.read().strip()
                    if not text:
                        text = f"No text returned for page {page_idx + 1}."

                    page_no = page_idx + 1
                    pages.append(f"--- Page {page_no} ---\n{text}\n")

                except Exception as e:
                    self.logger.error(
                        "Error extracting text from PDF %s page %d with DeepSeek-OCR: %s",
                        pdf_path,
                        page_idx + 1,
                        str(e),
                    )
                    traceback.print_exc()
                    continue
                finally:
                    if tmp_img:
                        try:
                            os.remove(tmp_img)
                        except Exception:
                            pass
                    if out_dir:
                        shutil.rmtree(out_dir, ignore_errors=True)

            return pages
        except Exception as e:
            self.logger.error(
                "Error extracting text from %s with DeepSeek-OCR: %s", pdf_path, str(e)
            )
            traceback.print_exc()
            return []
