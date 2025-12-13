import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import logging
import argparse
from pathlib import Path
from typing import Dict, List

from dots_ocr.parser import DotsOCRParser
from ocr_extractor import OCRExtractor
from qwenvl_caption import QwenVLCaptioner
from openai import OpenAI

from visdom import TextualRAGEngine
from dotenv import load_dotenv

load_dotenv()

class DummyTextParent:
    """Minimal parent object to drive TextualRAGEngine for testing.

    This parent uses a real PDF file and DotsOCR to build `document_cache`,
    then lets TextualRAGEngine run its normal OCR + embedding retrieval flow.
    """

    def __init__(self, pdf_path: str, text_retriever: str = "minilm", top_k: int = 5):
        # Config / basic attributes
        self.config: Dict[str, object] = {}
        self.data_dir = str(Path(pdf_path).parent)
        self.output_dir = str(Path(self.data_dir) / "visdom_test_output_text")
        os.makedirs(self.output_dir, exist_ok=True)

        self.llm_model = "dummy"
        self.api_keys: Dict[str, str] = {}

        # Use an embedding-based retriever to exercise the embedding path
        self.text_retriever = text_retriever
        self.vision_retriever = "nemo"
        self.top_k = top_k
        self.force_reindex = False
        self.qa_prompt = "Answer the question based on the document text."

        # We don't use benchmark CSV / df in this unit test
        self.df = None

        # Single test PDF
        self.pdf_files = [pdf_path]

        # Document cache: {doc_id: [page_text, ...]}
        self.document_cache: Dict[str, List[str]] = {}

    # --- Helpers expected by TextualRAGEngine ---

    def extract_text_from_pdf(self, pdf_path: str) -> List[str]:
        """Extract per-page text from a PDF using DotsOCR (similar to VisDoMRAG)."""

        dots_output_dir = Path(self.output_dir) / "dots_ocr_text_test"
        os.makedirs(dots_output_dir, exist_ok=True)

        dots_kwargs = {
            "max_completion_tokens": self.config.get("dots_max_completion_tokens", 4096),
            "num_thread": self.config.get("dots_num_thread", 64),
            "dpi": self.config.get("dots_dpi", 200),
            "output_dir": str(dots_output_dir),
        }

        parser = DotsOCRParser(**dots_kwargs)
        results = parser.parse_file(
            pdf_path,
            output_dir=str(dots_output_dir),
            prompt_mode=self.config.get("dots_prompt_mode", "prompt_layout_all_en"),
        )

        pages: List[str] = []
        for res in sorted(results, key=lambda r: r.get("page_no", 0)):
            md_path = res.get("md_content_path") or res.get("md_content_nohf_path")
            page_no = res.get("page_no", len(pages)) + 1
            page_text = ""
            if md_path and os.path.exists(md_path):
                with open(md_path, "r", encoding="utf-8") as f:
                    page_text = f.read()
            pages.append(f"--- Page {page_no} ---\n{page_text}\n")

        return pages

    def cache_documents(self) -> Dict[str, List[str]]:
        """Populate `document_cache` from the configured PDF file using OCR."""

        if self.document_cache:
            return self.document_cache

        for entry in self.pdf_files:
            pdf_path = Path(entry)
            if not pdf_path.exists():
                continue
            doc_id = pdf_path.stem
            self.document_cache[doc_id] = self.extract_text_from_pdf(str(pdf_path))

        return self.document_cache

    def split_text(self, text: str) -> List[str]:
        """Simple splitter: treat each page text as a single chunk."""
        return [text]



def test_textual_retrieval() -> None:
    """Basic sanity test for TextualRAGEngine interactive retrieval.

    It loads a real PDF file, runs DotsOCR to extract page texts into
    `document_cache`, builds an in-memory embedding index and runs a
    query, then prints out the retrieved text chunks.
    """

    pdf_path = os.environ.get(
        "VISDOM_TEXT_TEST_PDF",
        "/home/zechuan/m3docrag/contents/2024_Tencent_ESG.pdf",
    )

    if not os.path.exists(pdf_path):
        print(f"[SKIP] PDF file not found: {pdf_path}")
        return

    parent = DummyTextParent(pdf_path=pdf_path, text_retriever="minilm", top_k=5)

    try:
        engine = TextualRAGEngine(parent)
    except ImportError as e:
        print(f"[SKIP] TextualRAGEngine dependencies missing: {e}")
        return
    except Exception as e:
        print(f"[ERROR] Failed to initialize TextualRAGEngine: {e}")
        return

    query = "截至二零二四年每收入单位的温室气体排放总量是多少？"
    print(f"[INFO] Running textual retrieval for query: {query}")

    try:
        contexts = engine.retrieve_textual_contexts_for_question(query)
    except Exception as e:
        print(f"[ERROR] Textual retrieval raised an exception: {e}")
        return

    print(f"[RESULT] Retrieved {len(contexts)} textual contexts")
    for i, ctx in enumerate(contexts):
        chunk = ctx.get("chunk", "")
        doc_id = ctx.get("chunk_pdf_name")
        page_num = ctx.get("pdf_page_number")
        preview = chunk[:60].replace("\n", " ") + ("..." if len(chunk) > 60 else "")
        print(f"  - #{i}: doc={doc_id}, page={page_num}, text_preview={preview}")


class DummyChunkParent:
    def __init__(self, document_cache):
        self.config = {"text_chunk_size": 20, "text_chunk_overlap": 0}
        self.data_dir = "."
        self.output_dir = "."
        self.llm_model = "dummy"
        self.api_keys = {}
        self.text_retriever = "bm25"
        self.vision_retriever = "nemo"
        self.top_k = 5
        self.force_reindex = False
        self.qa_prompt = ""
        self.df = None
        self.pdf_files = []
        self.document_cache = document_cache

    def cache_documents(self):
        return self.document_cache


def test_chunk_routing_table_page() -> None:
    table_page = "--- Page 1 ---\n" "before\n" "<table><tr><td>a</td></tr></table>\n" "after\n"
    parent = DummyChunkParent({"doc_table": [table_page]})
    engine = TextualRAGEngine(parent)
    engine._build_interactive_text_index()
    assert len(engine._text_chunks) == 1
    assert engine._text_chunks[0] == table_page
    assert engine._text_chunk_mapping[0]["chunk_pdf_name"] == "doc_table"
    assert engine._text_chunk_mapping[0]["pdf_page_number"] == 0


def test_chunk_routing_image_page() -> None:
    image_block = "![](images/foo.png)\ncaption: foo bar"
    page = "--- Page 1 ---\n" "intro text\n" + image_block + "\nmore text\n"
    parent = DummyChunkParent({"doc_image": [page]})
    engine = TextualRAGEngine(parent)
    engine._build_interactive_text_index()
    chunks = engine._text_chunks
    mapping = engine._text_chunk_mapping
    assert len(chunks) >= 2
    matching_indices = [
        i
        for i, c in enumerate(chunks)
        if "images/foo.png" in c and "caption: foo bar" in c
    ]
    assert matching_indices
    idx = matching_indices[0]
    assert mapping[idx]["chunk_pdf_name"] == "doc_image"
    assert mapping[idx]["pdf_page_number"] == 0


def test_chunk_routing_plain_text_page() -> None:
    text = "--- Page 1 ---\n" + "This is a long plain text " * 20
    parent = DummyChunkParent({"doc_plain": [text]})
    engine = TextualRAGEngine(parent)
    engine._build_interactive_text_index()
    chunks = engine._text_chunks
    mapping = engine._text_chunk_mapping
    assert len(chunks) >= 2
    for m in mapping:
        assert m["chunk_pdf_name"] == "doc_plain"
        assert m["pdf_page_number"] == 0


class FullPipelineTextParent:
    """Parent that mirrors VisDoMRAG's OCR + captioning path for full-pipeline tests.

    This uses OCRExtractor (DotsOCR) to read a real PDF into per-page markdown,
    optionally augmenting images with Qwen-VL captions when QWEN_VL_SERVER_URL
    is configured, then exposes the same attributes that TextualRAGEngine expects.
    """

    def __init__(self, args: argparse.Namespace):
        self.config: Dict[str, object] = {
            "dots_max_completion_tokens": 4096,
            "dots_num_thread": 64,
            "dots_dpi": 200,
            "dots_prompt_mode": "prompt_layout_all_en",
            "qwen_vl_server_url": args.qwen_vl_server_url,
            "qwen_vl_model": args.qwen_vl_model,
            "qwen_vl_api_key": args.qwen_vl_api_key,
        }

        pdf_path = args.pdf_path

        self.data_dir = str(Path(pdf_path).parent)
        self.output_dir = str(Path(self.data_dir) / "visdom_full_text_test")
        os.makedirs(self.output_dir, exist_ok=True)

        self.llm_model = args.llm_model
        self.api_keys: Dict[str, str] = {
            "openai": "your-api-key",
            "doubao": args.doubao_api_key,
        }

        self.text_retriever = args.text_retriever
        self.vision_retriever = "nemo"
        self.top_k = args.top_k
        self.force_reindex = False
        self.qa_prompt = "Answer the question based on the document text."

        self.df = None
        self.pdf_files = [pdf_path]
        self.document_cache: Dict[str, List[str]] = {}

        # Reuse the same OCR + captioning helpers as VisDoMRAG.
        self._logger = logging.getLogger("VisDoMRAGTest")
        self.ocr_extractor = OCRExtractor(self.config, self.output_dir, self._logger)
        self.qwen_captioner = QwenVLCaptioner(
            self.config.get("qwen_vl_server_url"),
            self.config.get("qwen_vl_model", "Qwen/Qwen3-VL-4B-Instruct"),
            self._logger,
            self.config.get("qwen_vl_api_key"),
        )

        # Initialize the LLM for the generation step
        self._initialize_llm()

    def _initialize_llm(self):
        """Initialize the LLM based on the selected model."""
        if self.llm_model == "doubao":
            if not self.api_keys.get("doubao"):
                raise ValueError("Doubao API key is required for the test")
            self.llm = OpenAI(
                base_url="https://ark.cn-beijing.volces.com/api/v3",
                api_key=self.api_keys["doubao"],
            )
            self._logger.info("Initialized doubao model for testing")
        
        elif self.llm_model == "gpt4":
            if not self.api_keys.get("openai"):
                raise ValueError("OpenAI API key is required for the test")
            self.client = OpenAI(api_key=self.api_keys["openai"])
            self._logger.info("Initialized GPT-4 for testing")
        
        elif self.llm_model == "qwen":
            # For Qwen, the TextualRAGEngine expects the model handles to be on the parent.
            # This setup is more complex and typically loaded in the main VisDoMRAG class.
            # We will skip loading it here and assume it's handled if needed, or rely on other models for this test.
            self._logger.warning("Qwen model loading is not implemented in the test parent; use doubao or gpt4.")
        else:
            raise ValueError(f"Unsupported LLM model for testing: {self.llm_model}")

    def extract_text_from_pdf(self, pdf_path: str) -> List[str]:
        captioner = None
        if self.config.get("qwen_vl_server_url"):
            captioner = self.qwen_captioner
        return self.ocr_extractor.extract_text_from_pdf(
            pdf_path,
            ocr_engine="dots",
            captioner=captioner,
        )

    def cache_documents(self) -> Dict[str, List[str]]:
        if self.document_cache:
            return self.document_cache
        for entry in self.pdf_files:
            pdf_path = Path(entry)
            if not pdf_path.exists():
                continue
            doc_id = pdf_path.stem
            self.document_cache[doc_id] = self.extract_text_from_pdf(str(pdf_path))
        return self.document_cache


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
        print(f"[SKIP] PDF file not found for full pipeline test: {pdf_path}")
        return

    if not args.qwen_vl_server_url:
        print("[SKIP] QWEN_VL_SERVER_URL not set; skipping caption integration test")
        return

    parent = FullPipelineTextParent(args)

    try:
        engine = TextualRAGEngine(parent)
    except Exception as e:
        print(f"[ERROR] Failed to initialize TextualRAGEngine for full pipeline: {e}")
        return

    query = args.query
    print(f"[INFO] Running full textual pipeline for query: {query}")

    try:
        contexts = engine.retrieve_textual_contexts_for_question(query)
    except Exception as e:
        print(f"[ERROR] Full textual pipeline retrieval raised an exception: {e}")
        return

    print(f"[RESULT] Full pipeline retrieved {len(contexts)} textual contexts")
    for i, ctx in enumerate(contexts):
        chunk = ctx.get("chunk", "")
        doc_id = ctx.get("chunk_pdf_name")
        page_num = ctx.get("pdf_page_number")
        preview = chunk[:80].replace("\n", " ") + ("..." if len(chunk) > 80 else "")
        print(f"  - #{i}: doc={doc_id}, page={page_num}, text_preview={preview}")

    # Optionally, check whether any retrieved chunk contains an image caption
    # marker to give a quick signal that Qwen-VL augmentation is flowing
    # through the pipeline. We don't assert on this to avoid flakiness.
    has_caption = any("caption:" in ctx.get("chunk", "") for ctx in contexts)
    print(f"[INFO] Any 'caption:' marker present in retrieved chunks: {has_caption}")

    if not contexts:
        print("[SKIP] No contexts were retrieved, skipping generation step.")
        return

    print(f"\n[INFO] Generating textual response using LLM: {parent.llm_model}...")
    try:
        response = engine.generate_textual_response(query, contexts)
        print("--- LLM Response ---")
        print(response)
        print("--- End LLM Response ---")
        assert response and "Answer:" in response
    except Exception as e:
        print(f"[ERROR] Textual response generation raised an exception: {e}")
        return

def main(args: argparse.Namespace) -> None:
    # Determine which tests to run based on flags
    run_all = not (args.run_unit_tests or args.run_integration_test or args.run_full_pipeline_test)

    if run_all or args.run_unit_tests:
        print("\n--- Running Chunk Routing Unit Tests ---")
        test_chunk_routing_table_page()
        test_chunk_routing_image_page()
        test_chunk_routing_plain_text_page()
        print("--- Unit Tests Passed ---")

    if run_all or args.run_integration_test:
        print("\n--- Running Basic Integration Test ---")
        test_textual_retrieval()
        print("--- Basic Integration Test Finished ---")

    if run_all or args.run_full_pipeline_test:
        print("\n--- Running Full Pipeline Test (OCR + Captions) ---")
        test_full_textual_pipeline_with_ocr_and_captions(args)
        print("--- Full Pipeline Test Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tests for TextualRAGEngine.")
    parser.add_argument(
        "--pdf-path",
        type=str,
        default=os.environ.get(
            "VISDOM_TEXT_TEST_PDF",
            "/home/zechuan/m3docrag/contents/2024_Tencent_ESG.pdf",
        ),
        help="Path to the PDF file for integration tests.",
    )
    parser.add_argument(
        "--text-retriever",
        type=str,
        default="hybrid",
        choices=["minilm", "mpnet", "bge", "bm25", "hybrid"],
        help="Text retriever model to use.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Number of contexts to retrieve.")
    parser.add_argument(
        "--query",
        type=str,
        default="截至二零二四年每收入单位的温室气体排放总量是多少？",
        help="Query for retrieval tests.",
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
        "--run-unit-tests", action="store_true", help="Run only the chunk routing unit tests."
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
