import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import argparse
from pathlib import Path
from typing import Dict, List
import logging

from dotenv import load_dotenv
from visdom import VisDoMRAG

load_dotenv()
# Configure logging in the entry script so all VisDoMRAG loggers share it
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("test_textual_rag2.log"), logging.StreamHandler()],
    force=True,
)
logger = logging.getLogger("VisDoMRAG")

def build_visdom_for_pdf(pdf_path: List[str], args: argparse.Namespace) -> VisDoMRAG:
    """Construct a VisDoMRAG instance for a single-PDF textual test.

    This reuses VisDoMRAG's internal implementations of _initialize_llm,
    extract_text_from_pdf and cache_documents instead of re-implementing
    them in the test suite.
    """

    data_dir = str(Path(pdf_path[0]).parent)
    output_dir = str(Path(data_dir) / "visdom_text_test_output")
    os.makedirs(output_dir, exist_ok=True)

    config: Dict[str, object] = {
        "data_dir": data_dir,
        "output_dir": output_dir,
        "llm_model": args.llm_model,
        "vision_retriever": "nemo",
        "text_retriever": args.text_retriever,
        "top_k": args.top_k*len(pdf_path),
        "api_keys": {
            "openai": getattr(args, "openai_api_key", None),
            "doubao": args.doubao_api_key,
        },
        "force_reindex": False,
        "qa_prompt": "Answer the question based on the document text.",
        "pdf_files": pdf_path,
        "csv_path": None,
        "ocr_engine": "dots",
        "dots_max_completion_tokens": 4096,
        "dots_num_thread": 64,
        "dots_dpi": 200,
        "dots_prompt_mode": "prompt_layout_all_en",
        "qwen_vl_server_url": args.qwen_vl_server_url,
        "qwen_vl_model": args.qwen_vl_model,
        "qwen_vl_api_key": args.qwen_vl_api_key,
        "text_use_rerank": args.text_use_rerank,
    }

    return VisDoMRAG(config)



def test_textual_retrieval(args: argparse.Namespace) -> None:
    """Basic sanity test for TextualRAGEngine interactive retrieval.

    It loads a real PDF file, runs DotsOCR to extract page texts into
    `document_cache`, builds an in-memory embedding index and runs a
    query, then prints out the retrieved text chunks.
    """

    pdf_path = args.pdf_path or os.environ.get(
        "VISDOM_TEXT_TEST_PDF",
        "/home/zechuan/m3docrag/contents/2024_Tencent_ESG.pdf",
    )
    for pdf in pdf_path:
        if not os.path.exists(pdf):
            logger.info(f"[SKIP] PDF file not found: {pdf}")
            return

    try:
        visdom = build_visdom_for_pdf(pdf_path, args)
    except Exception as e:
        logger.info(f"[ERROR] Failed to initialize VisDoMRAG: {e}")
        return

    query = args.query or "截至二零二四年每收入单位的温室气体排放总量是多少？"
    logger.info(f"[INFO] Running textual retrieval for query: {query}")

    try:
        contexts = visdom.retrieve_textual_contexts_for_question(query)
    except Exception as e:
        logger.info(f"[ERROR] Textual retrieval raised an exception: {e}")
        return

    logger.info(f"[RESULT] Retrieved {len(contexts)} textual contexts")
    for i, ctx in enumerate(contexts):
        chunk = ctx.get("chunk", "")
        doc_id = ctx.get("chunk_pdf_name")
        page_num = ctx.get("pdf_page_number")
        preview = chunk[:60].replace("\n", " ") + ("..." if len(chunk) > 60 else "")
        logger.info(f"  - #{i}: doc={doc_id}, page={page_num}, text_preview={preview}")

def test_full_textual_pipeline_with_ocr_and_captions(args: argparse.Namespace) -> None:
    """Run a full textual pipeline: OCR + (optional) image captions + retrieval.

    This test uses a real PDF, runs it through OCRExtractor (DotsOCR) with
    optional Qwen-VL captioning, builds the interactive text index, and then
    runs an interactive textual retrieval query.

    It is designed as an integration test and will gracefully skip if either
    the test PDF is missing or caption server configuration is unavailable.
    """

    pdf_path = args.pdf_path
    if not pdf_path or not os.path.exists(pdf_path):
        logger.info(f"[SKIP] PDF file not found for full pipeline test: {pdf_path}")
        return

    if not args.qwen_vl_server_url:
        logger.info("[SKIP] QWEN_VL_SERVER_URL not set; skipping caption integration test")
        return

    try:
        visdom = build_visdom_for_pdf(pdf_path, args)
    except Exception as e:
        logger.info(f"[ERROR] Failed to initialize VisDoMRAG for full pipeline: {e}")
        return

    query = args.query
    logger.info(f"[INFO] Running full textual pipeline for query: {query}")

    try:
        contexts = visdom.retrieve_textual_contexts_for_question(query)
    except Exception as e:
        logger.info(f"[ERROR] Full textual pipeline retrieval raised an exception: {e}")
        return

    logger.info(f"[RESULT] Full pipeline retrieved {len(contexts)} textual contexts")
    for i, ctx in enumerate(contexts):
        chunk = ctx.get("chunk", "")
        doc_id = ctx.get("chunk_pdf_name")
        page_num = ctx.get("pdf_page_number")
        preview = chunk[:80].replace("\n", " ") + ("..." if len(chunk) > 80 else "")
        logger.info(f"  - #{i}: doc={doc_id}, page={page_num}, text_preview={preview}")

    # Optionally, check whether any retrieved chunk contains an image caption
    # marker to give a quick signal that Qwen-VL augmentation is flowing
    # through the pipeline. We don't assert on this to avoid flakiness.
    has_caption = any("caption:" in ctx.get("chunk", "") for ctx in contexts)
    logger.info(f"[INFO] Any 'caption:' marker present in retrieved chunks: {has_caption}")

    if not contexts:
        logger.info("[SKIP] No contexts were retrieved, skipping generation step.")
        return

    logger.info(f"\n[INFO] Generating textual response using LLM: {args.llm_model}...")
    try:
        response = visdom.generate_textual_response(query, contexts)
        logger.info("--- LLM Response ---")
        logger.info(response)
        logger.info("--- End LLM Response ---")
        assert response and "Answer:" in response
    except Exception as e:
        logger.info(f"[ERROR] Textual response generation raised an exception: {e}")
        return

def main(args: argparse.Namespace) -> None:
    # Determine which tests to run based on flags
    run_all = not (args.run_integration_test or args.run_full_pipeline_test)

    if run_all or args.run_integration_test:
        logger.info("\n--- Running Basic Integration Test ---")
        test_textual_retrieval(args)
        logger.info("--- Basic Integration Test Finished ---")

    if run_all or args.run_full_pipeline_test:
        logger.info("\n--- Running Full Pipeline Test (OCR + Captions) ---")
        test_full_textual_pipeline_with_ocr_and_captions(args)
        logger.info("--- Full Pipeline Test Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tests for TextualRAGEngine.")
    parser.add_argument(
        "--pdf-path",
        type=List[str],
        default=[
            "/home/zechuan/m3docrag/contents/2024_Tencent_ESG.pdf",
            # "/home/zechuan/m3docrag/contents/2024_sanqi_ESG.pdf",
            # "/home/zechuan/m3docrag/contents/2024_architecture_ESG.pdf",
        ],
        help="Path to the PDF file for integration tests.",
    )
    parser.add_argument(
        "--text-retriever",
        type=str,
        default="minilm",
        choices=["minilm", "mpnet", "bge", "bm25", "hybrid"],
        help="Text retriever model to use.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of contexts to retrieve.")
    parser.add_argument(
        "--query",
        type=str,
        default="公司的2024年男性员工数量分别是多少？", #"在反舞弊举报及调查中，包含哪些操作？",
        help="Query for retrieval tests.",
    )
    parser.add_argument(
        "--text_use_rerank",
        action="store_true",
        help="Enable CrossEncoder-based reranking for textual retrieval.",
    )
    parser.add_argument(
        "--qwen-vl-server-url",
        type=str,
        default=os.environ.get("QWEN_VL_SERVER_URL"),
        help="URL for the Qwen-VL vLLM server.",
    )
    parser.add_argument(
        "--qwen-vl-model",
        type=str,
        default=os.environ.get("QWEN_VL_MODEL", "Qwen/Qwen3-VL-4B-Instruct"),
        help="Model name for Qwen-VL captioning.",
    )
    parser.add_argument(
        "--qwen-vl-api-key",
        type=str,
        default=os.environ.get("QWEN_VL_API_KEY"),
        help="API key for the Qwen-VL server.",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="doubao",
        choices=["doubao", "gpt4", "qwen"],
        help="LLM to use for generating answers in the full pipeline test.",
    )
    parser.add_argument(
        "--doubao-api-key",
        type=str,
        default=os.getenv("ARK_API_KEY"),
        help="Doubao API key.",
    )
    parser.add_argument(
        "--run-integration-test",
        action="store_true",
        help="Run only the basic integration test.",
    )
    parser.add_argument(
        "--run-full-pipeline-test",
        action="store_true",
        help="Run only the full pipeline test with OCR and captions.",
    )

    parsed_args = parser.parse_args()
    main(parsed_args)
