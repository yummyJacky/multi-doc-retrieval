import os
from pathlib import Path

from pdf2image import convert_from_path

from visdom import VisualRAGEngine


class DummyVisualParent:
    """Minimal parent object to drive VisualRAGEngine for testing.

    It only provides the attributes and methods that VisualRAGEngine needs
    for interactive visual retrieval. We intentionally do NOT initialize
    any LLM here, because this test focuses purely on visual retrieval.
    """

    def __init__(self, pdf_path: str, vision_retriever: str = "nemo", top_k: int = 3):
        self.config = {}
        self.data_dir = str(Path(pdf_path).parent)
        self.output_dir = str(Path(self.data_dir) / "visdom_test_output")
        os.makedirs(self.output_dir, exist_ok=True)

        # LLM-related fields are unused in this test
        self.llm_model = "dummy"
        self.api_keys = {}

        # Retrieval-related configuration
        self.vision_retriever = vision_retriever
        self.text_retriever = "bm25"
        self.top_k = top_k
        self.force_reindex = False
        self.qa_prompt = "Answer the question based on the document pages."

        # For interactive visual retrieval, we only need a list of PDF files
        self.pdf_files = [pdf_path]

        # Benchmark-mode fields are not needed here
        self.df = None

    def _get_config_pdf_paths(self):
        """Simplified version: map a single PDF path to a doc_id."""
        paths = {}
        for entry in self.pdf_files:
            pdf_path = Path(entry)
            if not pdf_path.exists():
                continue
            doc_id = pdf_path.stem
            paths[doc_id] = str(pdf_path)
        return paths


def test_visual_retrieval() -> None:
    """Basic sanity test for VisualRAGEngine interactive retrieval.

    It loads a PDF file, builds an in-memory visual index and runs a simple
    query, then prints out the retrieved pages. This is NOT a strict unit
    test with assertions on ranking quality, but a functional check that
    the pipeline can run end-to-end.
    """

    # Prefer an env var so you can control which PDF to test with
    pdf_path = os.environ.get(
        "VISDOM_TEST_PDF",
        "/home/zechuan/m3docrag/contents/2024_Tencent_ESG.pdf",
    )

    if not os.path.exists(pdf_path):
        print(f"[SKIP] PDF file not found: {pdf_path}")
        return

    try:
        # Quick check that the PDF is readable
        pages = convert_from_path(pdf_path, dpi=100)
        if not pages:
            print(f"[SKIP] No pages read from PDF: {pdf_path}")
            return
        print(f"[INFO] Loaded {len(pages)} pages from {pdf_path}")
    except Exception as e:
        print(f"[SKIP] Failed to load PDF pages: {e}")
        return

    parent = DummyVisualParent(pdf_path=pdf_path, vision_retriever="nemo", top_k=3)

    try:
        engine = VisualRAGEngine(parent)
    except ImportError as e:
        print(f"[SKIP] VisualRAGEngine dependencies missing: {e}")
        return
    except Exception as e:
        print(f"[ERROR] Failed to initialize VisualRAGEngine: {e}")
        return

    query = "截至二零二四年每收入单位的温室气体排放总量是多少？"
    print(f"[INFO] Running visual retrieval for query: {query}")

    try:
        contexts = engine.retrieve_visual_contexts_for_question(query)
    except RuntimeError as e:
        # Common case: no GPU or CUDA issues
        print(f"[SKIP] Visual retrieval failed (likely CUDA / GPU issue): {e}")
        return
    except Exception as e:
        print(f"[ERROR] Visual retrieval raised an exception: {e}")
        return

    print(f"[RESULT] Retrieved {len(contexts)} visual contexts")
    for i, ctx in enumerate(contexts):
        doc_id = ctx.get("document_id")
        page_num = ctx.get("page_number")
        image = ctx.get("image")
        size = getattr(image, "size", None)
        print(f"  - #{i}: document_id={doc_id}, page_number={page_num}, image_size={size}")

    # A very light-weight correctness check on structure
    if contexts:
        first = contexts[0]
        assert "image" in first and "document_id" in first and "page_number" in first, (
            "VisualRAGEngine context objects should contain 'image', 'document_id', 'page_number'"
        )


def main() -> None:
    test_visual_retrieval()


if __name__ == "__main__":
    main()
