import os
from dotenv import load_dotenv
import argparse
import logging

from visdom import VisDoMRAG
load_dotenv()

def build_test_config(
    pdf_path: str,
    data_dir: str,
    output_dir: str,
    qwen_server_url: str,
    qwen_model: str,
    qwen_api_key,
):
    """Build a minimal config dict for testing DotsOCR + Qwen-VL caption pipeline."""
    pdf_path = os.path.abspath(pdf_path)
    data_dir = os.path.abspath(data_dir)
    output_dir = os.path.abspath(output_dir)

    return {
        "data_dir": data_dir,
        "output_dir": output_dir,
        "llm_model": "doubao",            # use qwen so Qwen2-VL text model is available if needed
        "vision_retriever": "nemo",       # not used in this test
        "text_retriever": "minilm",         # not used in this test
        "top_k": 5,
        "api_keys": {
            "openai": "your-openai-key",
            "doubao": os.getenv("ARK_API_KEY")
        },
        "chunk_size": 3000,
        "chunk_overlap": 300,
        "force_reindex": False,
        "qa_prompt": "",                  # not used
        "pdf_files": [pdf_path],
        "csv_path": None,
        "ocr_engine": "dots",             # explicitly use DotsOCR
        # DotsOCR settings (you can adjust if needed)
        "dots_max_completion_tokens": 4096,
        "dots_num_thread": 64,
        "dots_dpi": 200,
        "dots_prompt_mode": "prompt_layout_all_en",
        # Qwen-VL caption served by external vLLM HTTP server
        "qwen_vl_server_url": qwen_server_url,
        "qwen_vl_model": qwen_model,
        "qwen_vl_api_key": qwen_api_key,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test pipeline: DotsOCR parse PDF + Qwen-VL image caption to markdown.",
    )
    parser.add_argument("--pdf_path", type=str, help="Path to input PDF file")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./contents",
        help="Data directory (used only for consistency with VisDoMRAG config)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./contents/caption_test_output",
        help="Output directory for DotsOCR and test artifacts",
    )
    parser.add_argument(
        "--qwen_server_url",
        type=str,
        default="http://127.0.0.1:9000",
        help="Base URL of external Qwen-VL vLLM server, e.g. http://127.0.0.1:9000",
    )
    parser.add_argument(
        "--qwen_model",
        type=str,
        default="Qwen/Qwen3-VL-4B-Instruct",
        help="Model name served by vLLM (must match --served-model-name in vllm serve)",
    )
    parser.add_argument(
        "--qwen_api_key",
        type=str,
        default=None,
        help="Optional API key for the vLLM OpenAI server (if configured)",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("DotsOCR_QwenVL_Test")

    config = build_test_config(
        pdf_path=args.pdf_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        qwen_server_url=args.qwen_server_url,
        qwen_model=args.qwen_model,
        qwen_api_key=args.qwen_api_key,
    )

    logger.info("Initializing VisDoMRAG with test config...")
    vis = VisDoMRAG(config)

    logger.info("Running cache_documents() to trigger DotsOCR + Qwen-VL caption pipeline...")
    cache = vis.cache_documents()

    if not cache:
        logger.warning("No documents were cached. Check pdf_path and config paths.")
        return

    for doc_id, pages in cache.items():
        print("\n==== Document:", doc_id, "====")
        print("Total pages:", len(pages))
        for i, page_text in enumerate(pages, start=1):
            print(f"\n---- Page {i} preview ----")
            # Show only first 40 lines for brevity
            lines = page_text.splitlines()
            # Simple check for caption lines
            caption_lines = [ln for ln in lines if ln.strip().startswith("caption:")]
            if caption_lines:
                print("\n[INFO] Found caption lines on this page:")
                for cl in caption_lines:
                    print("  ", cl)
            else:
                print("\n[WARN] No caption lines found on this page.")
        break  # Only inspect the first document in this simple test


if __name__ == "__main__":
    main()
