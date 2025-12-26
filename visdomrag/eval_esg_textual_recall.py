import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm.auto import tqdm

from visdom import VisDoMRAG
from textual_rag import TextualRAGEngine

logger = logging.getLogger("VisDoMRAG")


ESG_DIR = Path(__file__).resolve().parent / "results" / "dots_ocr" / "2024_Tencent_ESG"
JSONL_PATH = ESG_DIR / "2024_Tencent_ESG_table_qa_qwen2_5vl.jsonl"
PDF_DEFAULT = Path(__file__).resolve().parent.parent / "contents" / "2024_Tencent_ESG.pdf"


def load_qa_samples(jsonl_path: Path) -> List[Dict]:
    """Load QA records from the ESG table QA JSONL file.

    Each line is a JSON object with at least:
    - doc: "2024_Tencent_ESG"
    - page_number: str
    - question: str
    - answer: str
    """
    samples: List[Dict] = []
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if not isinstance(rec, dict):
                continue
            q = str(rec.get("question", "")).strip()
            page_str = str(rec.get("page_number", "")).strip()
            doc = str(rec.get("doc", "")).strip() or "2024_Tencent_ESG"
            if not q or not page_str:
                continue
            try:
                page_num = int(page_str)
            except ValueError:
                # If page_number is not an integer, skip
                continue
            rec["_page_int"] = page_num
            rec["_doc_id"] = doc
            samples.append(rec)
    return samples


class ESGMdParent:
    """Minimal parent for TextualRAGEngine that uses existing MD pages.

    It builds document_cache directly from markdown files under ESG_DIR, so we
    don't need to rerun OCR. This corresponds to your "no OCR, use existing md" mode.
    """

    def __init__(self, text_retriever: str = "bm25", top_k: int = 5) -> None:
        self.config: Dict[str, object] = {
            "text_chunk_size": 500,
            "text_chunk_overlap": 100,
        }
        self.data_dir = str(ESG_DIR)
        self.output_dir = str(ESG_DIR / "eval_textual_recall")
        os.makedirs(self.output_dir, exist_ok=True)

        self.llm_model = "dummy"
        self.api_keys: Dict[str, str] = {}
        self.text_retriever = text_retriever
        self.vision_retriever = "nemo"
        self.top_k = top_k
        self.force_reindex = False
        self.qa_prompt = "Answer the question based on the document text."

        self.df = None
        self.pdf_files: List[str] = []
        self.document_cache: Dict[str, List[str]] = {}

    def cache_documents(self) -> Dict[str, List[str]]:
        if self.document_cache:
            return self.document_cache

        doc_id = "2024_Tencent_ESG"

        # Build a numeric page_no -> text mapping so that page index
        # (0-based) aligns with the `page_number` field in JSONL.
        page_texts: Dict[int, str] = {}

        # Use all markdown pages that match *_page_XXX.md or *_page_XXX_nohf.md
        md_files = list(ESG_DIR.glob("*_page_*.md"))
        if not md_files:
            logger.warning("No markdown pages found under %s", ESG_DIR)

        for md_file in md_files:
            name = md_file.name
            # Extract numeric page index from filename, e.g.
            # 2024_Tencent_ESG_page_110.md -> 110
            # 2024_Tencent_ESG_page_110_nohf.md -> 110
            import re

            m = re.search(r"page_(\d+)(?:_nohf)?\.md$", name)
            if not m:
                continue
            page_no = int(m.group(1))

            try:
                text = md_file.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            # Prefer the full markdown over _nohf when both exist for the same page.
            if name.endswith("_nohf.md") and page_no in page_texts:
                continue

            page_texts[page_no] = text

        # Convert to a list ordered by numeric page_no so that
        # enumerate(pages) index matches the JSONL page_number.
        pages: List[str] = []
        for _, text in sorted(page_texts.items(), key=lambda kv: kv[0]):
            pages.append(text)

        self.document_cache[doc_id] = pages
        return self.document_cache


def build_textual_engine_from_md(text_retriever: str = "bm25", top_k: int = 5) -> TextualRAGEngine:
    parent = ESGMdParent(text_retriever=text_retriever, top_k=top_k)
    engine = TextualRAGEngine(parent)
    return engine


def build_textual_engine_from_pdf(pdf_path: Path, text_retriever: str = "bm25", top_k: int = 5) -> TextualRAGEngine:
    """Build TextualRAGEngine by running the full VisDoMRAG OCR pipeline.

    This corresponds to the "from OCR" mode.
    """
    config: Dict[str, object] = {
        "data_dir": str(pdf_path.parent),
        "output_dir": str(pdf_path.parent / "visdom_esg_eval_output"),
        "llm_model": "doubao",  # placeholder, we only care about retrieval
        "vision_retriever": "nemo",
        "text_retriever": text_retriever,
        "top_k": top_k,
        "api_keys": {},
        "pdf_files": [str(pdf_path)],
        "ocr_engine": "dots",
    }
    os.makedirs(config["output_dir"], exist_ok=True)

    visdom = VisDoMRAG(config)
    # Ensure documents are cached
    visdom.cache_documents()
    engine = visdom.textual_engine
    return engine


def eval_recall(engine: TextualRAGEngine, samples: List[Dict]) -> Tuple[float, Dict[int, int]]:
    """Evaluate page-level recall for ESG QA samples.

    For each question, we call `retrieve_textual_contexts_for_question(question)`
    and check whether any returned context has matching doc_id and page_number.

    Returns:
        recall: overall hit_rate (hits / total)
        hit_ranks: histogram of the (1-based) rank where the first hit appears
    """
    total = 0
    hits = 0
    hit_ranks: Dict[int, int] = {}

    for rec in tqdm(samples, desc="Evaluating textual recall"):
        question = rec["question"]
        target_page = rec["_page_int"]
        doc_id = rec["_doc_id"]

        total += 1
        try:
            contexts = engine.retrieve_textual_contexts_for_question(question)
        except Exception as e:
            logger.error("Error retrieving contexts for question: %s", e)
            continue

        if not contexts:
            continue

        hit_rank = None
        for idx, ctx in enumerate(contexts):
            ctx_doc = ctx.get("chunk_pdf_name") or doc_id
            ctx_page = ctx.get("pdf_page_number")
            try:
                ctx_page_int = int(ctx_page)
            except Exception:
                continue
            if ctx_doc == doc_id and ctx_page_int == target_page:
                hit_rank = idx + 1  # 1-based
                break

        if hit_rank is not None:
            hits += 1
            hit_ranks[hit_rank] = hit_ranks.get(hit_rank, 0) + 1

    recall = hits / total if total > 0 else 0.0
    return recall, hit_ranks


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate textual retrieval page recall on ESG table QA dataset.")
    parser.add_argument("--mode", choices=["ocr", "md"], default="md", help="Retrieval backend: 'ocr' runs full OCR, 'md' uses existing markdown pages.")
    parser.add_argument("--text-retriever", default="bm25", choices=["bm25", "minilm", "mpnet", "bge", "hybrid"], help="Text retriever type used by TextualRAGEngine.")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k contexts to retrieve.")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on number of QA samples to evaluate (0 means all).")
    parser.add_argument("--pdf-path", type=str, default=str(PDF_DEFAULT), help="Path to ESG PDF (used in 'ocr' mode).")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    samples = load_qa_samples(JSONL_PATH)
    if args.limit and args.limit > 0:
        samples = samples[: args.limit]

    if args.mode == "ocr":
        pdf_path = Path(args.pdf_path)
        if not pdf_path.is_file():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        engine = build_textual_engine_from_pdf(pdf_path, text_retriever=args.text_retriever, top_k=args.top_k)
    else:
        engine = build_textual_engine_from_md(text_retriever=args.text_retriever, top_k=args.top_k)

    recall, hit_ranks = eval_recall(engine, samples)

    print("==== Textual Retrieval Page Recall (ESG Table QA) ====")
    print(f"Total samples: {len(samples)}")
    print(f"Hits: {sum(hit_ranks.values())}")
    print(f"Recall@{args.top_k}: {recall:.4f}")
    if hit_ranks:
        print("Hit rank distribution (rank: count):")
        for rank in sorted(hit_ranks.keys()):
            print(f"  {rank}: {hit_ranks[rank]}")


if __name__ == "__main__":
    main()
