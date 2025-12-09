import os
from pathlib import Path
from typing import Dict, List

from dots_ocr.parser import DotsOCRParser

from visdom import TextualRAGEngine


class DummyTextParent:
    """Minimal parent object to drive TextualRAGEngine for testing.

    This parent uses a real PDF file and DotsOCR to build `document_cache`,
    then lets TextualRAGEngine run its normal OCR + embedding retrieval flow.
    """

    def __init__(self, pdf_path: str, text_retriever: str = "minilm", top_k: int = 3):
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

    parent = DummyTextParent(pdf_path=pdf_path, text_retriever="minilm", top_k=3)

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

def main() -> None:
    test_textual_retrieval()


if __name__ == "__main__":
    main()
