import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import asyncio
import requests
from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from deepeval.evaluate.configs import AsyncConfig
from openai import OpenAI
from tqdm import tqdm
from eval_esg_textual_recall import (
    ESG_DIR,
    JSONL_PATH,
    load_qa_samples,
)
from generate_esg_qa import QWEN_VL_API_KEY, _chat_completions_url
from textual_rag import TextualRAGEngine
from dotenv import load_dotenv

load_dotenv()  

logger = logging.getLogger("VisDoMRAG")

# 缓存豆包回答和检索上下文的 JSONL 文件
CACHE_DIR: Path = ESG_DIR / "eval_rag_deepeval"
CACHE_FILE: Path = CACHE_DIR / "rag_answer_cache.jsonl"
os.makedirs(CACHE_DIR, exist_ok=True)
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_NAME = DEFAULT_MODEL_NAME

class ESGMdParentDoubao:
    """Minimal parent for TextualRAGEngine using existing MD pages and Doubao LLM.

    - 文本来源：直接复用 ESG markdown 页面（不触发 OCR）。
    - LLM：初始化豆包 HTTP 客户端，用于 generate_textual_response。
    """

    def __init__(self, text_retriever: str = "bm25", top_k: int = 5) -> None:
        self.config: Dict[str, object] = {
            "text_chunk_size": 500,
            "text_chunk_overlap": 100,
        }
        self.data_dir = str(ESG_DIR)
        self.output_dir = str(ESG_DIR / "eval_rag_deepeval")
        os.makedirs(self.output_dir, exist_ok=True)

        self.llm_model = "doubao"
        self.api_keys: Dict[str, str] = {}

        doubao_key = os.getenv("ARK_API_KEY", "").strip()
        if not doubao_key:
            raise RuntimeError(
                "DOUBAO_API_KEY is not set; please export your Doubao API key before running evaluation."
            )
        self.api_keys["doubao"] = doubao_key

        # 初始化豆包 OpenAI 客户端（与 visdom.VisDoMRAG._initialize_llm 中逻辑一致）
        self.llm = OpenAI(
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key=doubao_key,
        )

        self.text_retriever = text_retriever
        self.vision_retriever = "nemo"
        self.top_k = top_k
        self.force_reindex = False
        self.qa_prompt = "Answer the question objectively based on the context provided."

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
            logger.warning("[ESGMdParentDoubao] No markdown pages found under %s", ESG_DIR)
        else:
            logger.info("[ESGMdParentDoubao] Found %d markdown pages under %s", len(md_files), ESG_DIR)

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
                logger.info("[ESGMdParentDoubao] Loaded page %d from %s", page_no, md_file)
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

        logger.info(
            "[ESGMdParentDoubao] Loaded %d pages into document_cache for doc_id=%s",
            len(pages),
            doc_id,
        )

        self.document_cache[doc_id] = pages
        return self.document_cache


def _build_rag_prompt(question: str, contexts: List[str]) -> str:
    """构造给 Qwen 文本接口的 RAG Prompt（仅文本，不重新 OCR）。"""
    qa_prompt = "Answer the question objectively based on the context provided."
    joined_contexts = "\n- ".join(c.strip() for c in contexts if c and c.strip())

    prompt = f"""
            You are tasked with answering a question based on the relevant chunks of a PDF document. Provide your response in the following format:
            ## Evidence:

            ## Chain of Thought:

            ## Answer:

            ___
            Instructions:

            1. Evidence Curation: Extract relevant elements (such as paragraphs, tables, figures, charts) from the provided chunks and populate them in the "Evidence" section. For each element, include the type, content, and a brief explanation of its relevance.

            2. Chain of Thought: In the "Chain of Thought" section, list out each logical step you take to derive the answer, referencing the evidence where applicable. You should perform computations if you need to to get to the answer. 

            3. Answer: {qa_prompt}
            ___
            Question: {question}
            ___
            Context: {joined_contexts}
            
            """
    return prompt


    # prompt = (
    #     "你是一名 ESG 报告分析助手。现在给你若干从 ESG 报告中检索到的文本片段，"
    #     "请严格基于这些片段来回答用户问题，不要编造报告中不存在的信息。\n\n"
    #     "回答要求：\n"
    #     "1. 必须基于给定的上下文，尽量引用其中的关键数字或结论；\n"
    #     "2. 如果上下文不足以回答问题，请明确说明“根据当前检索到的内容无法确定”；\n"
    #     "3. 回答用中文，语言简洁准确。\n\n"
    #     f"用户问题：{question}\n\n"
    #     f"检索到的上下文片段：\n- {joined_contexts}"
    # )
    return prompt


def _call_qwen_chat(prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
    """调用本地 Qwen2.5-VL HTTP 服务的纯文本 chat 接口。

    复用 generate_esg_qa.py 中的 `_chat_completions_url` / MODEL_NAME / QWEN_VL_API_KEY
    配置，只发送文本消息，不附带图片，从而作为评测用的 LLM。"""

    url = _chat_completions_url()
    if url is None:
        raise RuntimeError(
            "QWEN_VL_SERVER_URL is not set; cannot call Qwen-VL service for evaluation."
        )

    payload: Dict[str, object] = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    headers = {"Content-Type": "application/json"}
    if QWEN_VL_API_KEY:
        headers["Authorization"] = f"Bearer {QWEN_VL_API_KEY}"

    response = requests.post(
        url,
        headers=headers,
        data=json.dumps(payload, ensure_ascii=False),
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()

    choices = data.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message", {})
    content = message.get("content")

    # vLLM OpenAI server 可能返回字符串或 block 列表
    if isinstance(content, str):
        generated = content
    else:
        parts: List[str] = []
        for block in content or []:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        generated = "".join(parts)

    return (generated or "").strip()


class QwenLocalEvalModel(DeepEvalBaseLLM):
    """DeepEval 自定义 LLM，使用本地 Qwen2.5-VL 服务进行打分。"""

    def load_model(self, *args, **kwargs) -> "QwenLocalEvalModel":
        # 对于 HTTP 服务，不需要在本地加载权重，直接返回 self 即可。
        return self

    def generate(self, prompt: str, *args, **kwargs) -> str:  # type: ignore[override]
        return _call_qwen_chat(prompt)

    async def a_generate(self, prompt: str, *args, **kwargs) -> str:  # type: ignore[override]
        # 使用线程池异步执行阻塞的 HTTP 请求，以便真正并发多个评估调用
        return await asyncio.to_thread(_call_qwen_chat, prompt)

    def get_model_name(self, *args, **kwargs) -> str:  # type: ignore[override]
        # 使用 DeepEvalBaseLLM 默认的 name，如果缺失则回退到我们配置的 MODEL_NAME
        return getattr(self, "name", None) or MODEL_NAME


def build_rag_engine(text_retriever: str = "bm25", top_k: int = 5):
    """使用已有 markdown 页面构建 TextualRAGEngine，不重新 OCR，且启用豆包 LLM。

    同时在评测开始前，对所有 MD 页面执行一次分块 + 向量化建索引，
    避免交互式首次调用时才构建索引导致问题不易发现。
    """

    parent = ESGMdParentDoubao(text_retriever=text_retriever, top_k=top_k)

    # 1) 预先构建 document_cache
    doc_cache = parent.cache_documents()
    if not doc_cache:
        raise RuntimeError(
            "ESGMdParentDoubao.cache_documents() returned empty document_cache; "
            "please check ESG_DIR and markdown pages."
        )

    # 2) 初始化 TextualRAGEngine 并显式构建交互式文本索引
    engine = TextualRAGEngine(parent)
    engine._build_interactive_text_index()

    # 3) 简单检查 chunk 数
    text_chunks = getattr(engine, "_text_chunks", None)
    if not text_chunks:
        raise RuntimeError(
            "TextualRAGEngine built interactive text index with 0 chunks; "
            "verify that markdown pages contain non-empty text content."
        )

    logger.info(
        "[build_rag_engine] Built interactive text index with %d chunks",
        len(text_chunks),
    )

    return engine


def run_rag(
    question: str,
    engine,
) -> Tuple[str, List[str]]:
    """只使用 TextualRAGEngine 的文本检索 + 豆包模型生成回答。

    - 不会重新运行 OCR，只依赖 markdown 文本索引。
    - 检索后直接调 TextualRAGEngine.generate_textual_response（内部用豆包）。
    - 返回模型回答和用于回答的文本上下文列表。
    """

    contexts = engine.retrieve_textual_contexts_for_question(question) or []
    context_texts = [str(c.get("chunk", "")) for c in contexts if c.get("chunk")]

    # 豆包回答，prompt 模板与 textual_rag.generate_textual_response 中一致
    answer = engine.generate_textual_response(question, contexts)
    return answer, context_texts


def load_eval_dataset(limit: int = 0) -> List[Dict]:
    """复用 eval_esg_textual_recall.py 中的 ESG QA 数据加载逻辑。"""

    samples = load_qa_samples(JSONL_PATH)
    if limit > 0:
        samples = samples[:limit]
    return samples


def build_test_cases(engine, samples: List[Dict]) -> List[LLMTestCase]:
    test_cases: List[LLMTestCase] = []

    # 1) 先从缓存文件加载已存在的回答
    cache: Dict[str, Dict[str, object]] = {}
    if CACHE_FILE.is_file():
        try:
            with CACHE_FILE.open("r", encoding="utf-8") as f:
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
                    if not q:
                        continue
                    cache[q] = rec
            logger.info("[build_test_cases] Loaded %d cached answers from %s", len(cache), CACHE_FILE)
        except Exception as e:
            logger.error("[build_test_cases] Failed to load cache from %s: %s", CACHE_FILE, e)

    for rec in tqdm(samples, desc="Building test cases..."):
        question = str(rec.get("question", "")).strip()
        expected_answer = str(rec.get("answer", "")).strip()
        if not question:
            continue

        # 2) 优先使用缓存中的回答
        cached = cache.get(question)
        if cached is not None:
            answer = str(cached.get("answer", ""))
            contexts = cached.get("contexts", []) or []
        else:
            # 3) 若缓存中不存在，则调用一次 RAG，并把结果写入缓存
            answer, contexts = run_rag(question, engine)
            cache_entry = {
                "question": question,
                "answer": answer,
                "contexts": contexts,
            }
            cache[question] = cache_entry
            try:
                with CACHE_FILE.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(cache_entry, ensure_ascii=False) + "\n")
            except Exception as e:
                logger.error("[build_test_cases] Failed to append cache for question: %s", e)

        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            expected_output=expected_answer,
            retrieval_context=contexts,
            metadata={
                "doc": rec.get("doc"),
                "page_number": rec.get("page_number"),
                "image_path": rec.get("image_path"),
            },
        )
        test_cases.append(test_case)

    return test_cases


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate ESG RAG pipeline with DeepEval using Qwen2.5-VL and textual retrieval only.",
    )
    parser.add_argument(
        "--text-retriever",
        default="bm25",
        choices=["bm25", "minilm", "mpnet", "bge", "hybrid"],
        help="Text retriever type used by TextualRAGEngine.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-k contexts to retrieve for each question.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit on number of QA samples to evaluate (0 means all).",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=64,
        help=(
            "Maximum number of concurrent evaluation requests sent to the Qwen "
            "evaluation model (DeepEval AsyncConfig.max_concurrent)."
        ),
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    print(f"Using ESG directory: {ESG_DIR}")
    print(f"Loading QA dataset from: {JSONL_PATH}")

    samples = load_eval_dataset(limit=args.limit)
    print(f"Loaded {len(samples)} QA samples for evaluation.")

    engine = build_rag_engine(text_retriever=args.text_retriever, top_k=args.top_k)

    # 构建 DeepEval 测试用例
    test_cases = build_test_cases(engine, samples)

    # 使用本地 Qwen2.5-VL 作为 metrics 所依赖的评估 LLM
    eval_model = QwenLocalEvalModel(model=MODEL_NAME)

    answer_relevancy = AnswerRelevancyMetric(
        model=eval_model,
        threshold=0.5,
        include_reason=True,
    )
    faithfulness = FaithfulnessMetric(
        model=eval_model,
        threshold=0.5,
        include_reason=True,
    )
    contextual_relevancy = ContextualRelevancyMetric(
        model=eval_model,
        threshold=0.5,
        include_reason=True,
    )

    print("Running DeepEval metrics (this may take a while)...")

    async_config = AsyncConfig(
        run_async=True,
        max_concurrent=max(1, args.max_concurrent),
    )

    result = evaluate(
        test_cases=test_cases,
        metrics=[answer_relevancy, faithfulness, contextual_relevancy],
        async_config=async_config,
    )

    print("===== DeepEval ESG RAG Evaluation Result =====")
    print(result)


if __name__ == "__main__":
    main()
