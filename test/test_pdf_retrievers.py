import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import sys
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)


sys.path.insert(0, project_root)

from colQwen_rag_with_faiss_pdf import (
    ColQwenPDFRetriever,
    GMELayoutPDFRetriever,
    GMETextPDFRetriever,
)
from models.gme import GmeQwen2VL
from pdf2image import convert_from_path

def load_pdf_images() -> List[Image.Image]:
    """加载用于测试的 PDF 页面图像"""
    pdf_path = "/home/zechuan/m3docrag/contents/2024_Tencent_ESG.pdf"

    if not os.path.exists(pdf_path):
        print(f"[SKIP] PDF file not found: {pdf_path}")
        return []

    images = convert_from_path(pdf_path, dpi=200)
    print(f"[INFO] Loaded {len(images)} pages from {pdf_path}")
    return images


def load_gme_model() -> Optional[GmeQwen2VL]:
    """加载 GME 模型（用于 Layout 和 Text 两个 retriever）"""
    gme_path = "Alibaba-NLP/gme-Qwen2-VL-7B-Instruct"
    if not gme_path:
        print("[SKIP] GME_PATH not set, skip GME retriever tests.")
        return None

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Loading GME model from {gme_path} on {device}")
    model = GmeQwen2VL(model_path=gme_path, device=device)
    return model


def test_colqwen_pdf_retriever(images: List[Image.Image]) -> None:
    """测试 ColQwenPDFRetriever 的基本检索功能"""
    colqwen_path = "vidore/colqwen2-v1.0"

    if not colqwen_path:
        print("[SKIP] COLQWEN2_7B_PATH not set, skip ColQwenPDFRetriever test.")
        return

    if not images:
        print("[SKIP] No images loaded, skip ColQwenPDFRetriever test.")
        return

    print(f"[INFO] Testing ColQwenPDFRetriever with model at {colqwen_path}")
    retriever = ColQwenPDFRetriever(colqwen_path=colqwen_path)

    query = "2022年员工总数是多少？"
    results = retriever.search(query, images, top_k=5)

    print(f"[RESULT] ColQwenPDFRetriever returned {len(results)} results")
    for r in results:
        print(
            f"  page_idx={r['page_idx']}, page_num={r['page_num']}, score={r['score']:.4f}"
        )


def test_gme_layout_pdf_retriever(gme_model: Optional[GmeQwen2VL], images: List[Image.Image]) -> None:
    """测试 GMELayoutPDFRetriever 的布局级检索功能"""
    if gme_model is None:
        print("[SKIP] No GME model, skip GMELayoutPDFRetriever test.")
        return

    if not images:
        print("[SKIP] No images loaded, skip GMELayoutPDFRetriever test.")
        return

    print("[INFO] Testing GMELayoutPDFRetriever")
    retriever = GMELayoutPDFRetriever(gme_model)
    retriever.build_index(images)

    query = "包含温室气体排放表格的页面"
    results = retriever.search(query, top_k=5)

    print(f"[RESULT] GMELayoutPDFRetriever returned {len(results)} results")
    for r in results:
        print(
            f"  page_idx={r['page_idx']}, page_num={r['page_num']}, score={r['score']:.4f}"
        )


def test_gme_text_pdf_retriever(gme_model: Optional[GmeQwen2VL], images: List[Image.Image]) -> None:
    """测试 GMETextPDFRetriever 的文本到文本检索功能（使用手动提供的文本，避免依赖 OCR）"""
    if gme_model is None:
        print("[SKIP] No GME model, skip GMETextPDFRetriever test.")
        return

    if not images:
        print("[SKIP] No images loaded, skip GMETextPDFRetriever test.")
        return

    print("[INFO] Testing GMETextPDFRetriever (without OCR, using provided texts)")
    retriever = GMETextPDFRetriever(gme_model, use_ocr=False)

    # 为每一页构造简单的示例文本，长度与图片页数对齐
    example_texts: List[str] = []
    for idx in range(len(images)):
        example_texts.append(f"This is example text for page {idx + 1} about greenhouse gas emissions and ESG report.")

    retriever.build_index(images, page_texts=example_texts)

    query = "温室气体排放"  # 一个与示例文本相关的查询
    results = retriever.search(query, top_k=5)

    print(f"[RESULT] GMETextPDFRetriever returned {len(results)} results")
    for r in results:
        print(
            f"  page_idx={r['page_idx']}, page_num={r['page_num']}, score={r['score']:.4f}"
        )


def main() -> None:
    images = load_pdf_images()

    # 分别测试三个 retriever
    # test_colqwen_pdf_retriever(images)

    gme_model = load_gme_model()
    test_gme_layout_pdf_retriever(gme_model, images)
    # test_gme_text_pdf_retriever(gme_model, images)


if __name__ == "__main__":
    main()
