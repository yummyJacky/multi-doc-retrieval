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

    # 1) 使用手动提供的文本进行文本到文本检索（不依赖 OCR）
    print("[INFO] Testing GMETextPDFRetriever (manual texts, use_ocr=False)")
    retriever = GMETextPDFRetriever(gme_model, use_ocr=False)

    # 为前几页构造内容差异明显的示例文本，便于检查检索结果是否合理
    num_pages = len(images)
    example_texts: List[str] = []
    for idx in range(num_pages):
        page_num = idx + 1
        if idx == 0:
            example_texts.append(
                f"第{page_num}页主要介绍公司的温室气体排放情况以及减排目标。"
            )
        elif idx == 1:
            example_texts.append(
                f"第{page_num}页重点描述公司的ESG管治架构和董事会职责。"
            )
        else:
            example_texts.append(
                f"第{page_num}页为一般性的ESG报告内容示例文本。"
            )

    retriever.build_index(images, page_texts=example_texts)

    query = "温室气体排放"  # 与第 1 页示例文本强相关的查询
    results = retriever.search(query, top_k=5)

    print(f"[RESULT] GMETextPDFRetriever (manual texts) returned {len(results)} results")
    for r in results:
        print(
            f"  page_idx={r['page_idx']}, page_num={r['page_num']}, score={r['score']:.4f}"
        )

def test_gme_text_pdf_retriever_with_ocr(gme_model: Optional[GmeQwen2VL], images: List[Image.Image]) -> None:
    if gme_model is None:
        print("[SKIP] No GME model, skip GMETextPDFRetriever (OCR) test.")
        return

    if not images:
        print("[SKIP] No images loaded, skip GMETextPDFRetriever (OCR) test.")
        return

    # 直接使用 OCR（DotOCR）抽取文本，再做文本到文本检索
    # 若环境或权重未配置好，仅打印 warning 不中断整体测试脚本
    try:
        print("[INFO] Testing GMETextPDFRetriever with OCR (DotOCR, use_ocr=True)")
        ocr_retriever = GMETextPDFRetriever(gme_model, use_ocr=True)

        start_page = 0
        ocr_images = images[start_page:]
        ocr_retriever.build_index(ocr_images)

        query = "温室气体指标在2030年的目标是什么?"
        ocr_results = ocr_retriever.search(query, top_k=5)
        print(
            f"[RESULT] GMETextPDFRetriever (DotOCR) returned {len(ocr_results)} results"
        )
        for r in ocr_results:
            print(
                f"  [OCR] page_idx={r['page_idx']+start_page}, page_num={r['page_num']+start_page}, score={r['score']:.4f}"
            )
    except Exception as e:
        print(f"[WARN] OCR-based GMETextPDFRetriever test failed: {e}")


def main() -> None:
    images = load_pdf_images()

    # 分别测试三个 retriever
    # test_colqwen_pdf_retriever(images)

    gme_model = load_gme_model()
    # test_gme_layout_pdf_retriever(gme_model, images)
    # test_gme_text_pdf_retriever(gme_model, images)
    test_gme_text_pdf_retriever_with_ocr(gme_model, images)


if __name__ == "__main__":
    main()
