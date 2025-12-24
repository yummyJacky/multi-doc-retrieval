import logging
import os
import traceback
import csv
import uuid
import re
from typing import List, Dict, Tuple, Optional
import chromadb
import pandas as pd
import torch
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter, Language
from tqdm.auto import tqdm
from sentence_transformers import CrossEncoder
from dataclasses import dataclass

logger = logging.getLogger("VisDoMRAG")

try:
    import cn2an  
    _HAS_CN2AN = True
except Exception:  
    cn2an = None  
    logger.warning("cn2an not installed, will not be able to convert Chinese numerals")
    _HAS_CN2AN = False

def _convert_chinese_numeral_token(token: str) -> str:
    if not token:
        return token

    # Prefer using cn2an if available for robust Chinese numeral parsing
    if _HAS_CN2AN:
        try:
            value = cn2an.cn2an(token, "smart")
            return str(value)
        except Exception:
            logger.warning("Failed to convert Chinese numeral token: %s", token)
            pass

    return token


_CHINESE_NUMERAL_PATTERN = re.compile(r"[零〇一二三四五六七八九]+")


def _normalize_chinese_numerals(text: str) -> str:
    if not text:
        return text

    def _repl(match):
        token = match.group(0)
        return _convert_chinese_numeral_token(token)

    return _CHINESE_NUMERAL_PATTERN.sub(_repl, text)


class SimpleBM25:
    def __init__(self, documents):
        self.documents = documents or []
        self._build_index()

    def _tokenize(self, text):
        import re
        import jieba

        chinese_pattern = r"[\u4e00-\u9fff]+"
        chinese_parts = re.findall(chinese_pattern, text)
        non_chinese_text = re.sub(chinese_pattern, " ", text)

        tokens = []

        for chinese_part in chinese_parts:
            chinese_tokens = list(jieba.cut(chinese_part, cut_all=False))
            tokens.extend([token.strip() for token in chinese_tokens if token.strip()])

        english_tokens = re.findall(r"[a-zA-Z]+|\d+", non_chinese_text.lower())
        tokens.extend(english_tokens)

        filtered_tokens = []
        for token in tokens:
            if token and (len(token) > 1 or token.isdigit()):
                filtered_tokens.append(token)

        return filtered_tokens

    def _build_index(self):
        from collections import defaultdict, Counter

        self.keyword_index = defaultdict(list)
        self.doc_lengths = []

        for i, doc in enumerate(self.documents):
            content = doc or ""
            tokens = self._tokenize(content)
            self.doc_lengths.append(len(tokens))

            token_counts = Counter(tokens)
            for token, count in token_counts.items():
                self.keyword_index[token].append((i, count))

        self.N = len(self.documents)
        self.avg_doc_length = (
            sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        )

    def get_scores(self, query):
        import math

        if not self.documents:
            return []

        if isinstance(query, str):
            query_tokens = self._tokenize(query)
        else:
            query_tokens = [t for t in query if t]

        if not query_tokens:
            return [0.0] * self.N

        k1, b = 1.5, 0.75
        scores = [0.0] * self.N
        avg_dl = self.avg_doc_length or 1.0

        for token in query_tokens:
            if token not in self.keyword_index:
                continue

            doc_entries = self.keyword_index[token]
            df = len(doc_entries)
            idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)

            for doc_idx, tf in doc_entries:
                doc_len = self.doc_lengths[doc_idx]
                denom = tf + k1 * (1 - b + b * doc_len / avg_dl)
                score = idf * (tf * (k1 + 1)) / denom
                scores[doc_idx] += score

        return scores


class TextualRAGEngine:
    def __init__(self, parent):
        """Engine responsible for textual retrieval and textual QA."""
        self.parent = parent
        self.config = parent.config
        self.data_dir = parent.data_dir
        self.output_dir = parent.output_dir
        self.llm_model = parent.llm_model
        self.text_retriever = parent.text_retriever
        self.top_k = parent.top_k
        self.force_reindex = parent.force_reindex
        self.qa_prompt = parent.qa_prompt
        self.df = parent.df
        self.api_keys = parent.api_keys

        # Copy LLM-related handles from parent
        if hasattr(parent, "client"):
            self.client = parent.client
        if hasattr(parent, "llm"):
            self.llm = parent.llm
        if hasattr(parent, "qwen_model"):
            self.qwen_model = parent.qwen_model
        if hasattr(parent, "qwen_processor"):
            self.qwen_processor = parent.qwen_processor

        # Text retrieval resources
        self.text_retrieval_file = f"{self.data_dir}/retrieval/retrieval_{self.text_retriever}.csv"

        if self.text_retriever == "bm25":
            # No model needed for BM25
            print("Using BM25 for text retrieval")
            self.st_embedding_function = None
        elif self.text_retriever in ["minilm", "mpnet", "bge", "hybrid"]:
            # Map text_retriever to actual model names
            print(f"Using {self.text_retriever} for text retrieval")
            # 测试看minilm选用多语言这个模型效果也不好，中文不如用bge
            model_map = {
                "minilm": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",#"sentence-transformers/all-MiniLM-L6-v2",
                "mpnet": "sentence-transformers/all-mpnet-base-v2",
                "bge": "BAAI/bge-large-zh-v1.5",
                # Hybrid uses MiniLM as the dense encoder
                "hybrid": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            }

            # Load sentence transformer model
            self.text_model_name = model_map[self.text_retriever]
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.st_embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.text_model_name, device=self.device
            )
        else:
            raise ValueError(f"Unsupported text retriever: {self.text_retriever}")

        # Optional CrossEncoder-based reranking
        self.use_rerank = self.config.get("text_use_rerank", False)
        # How many candidates to send to the CrossEncoder for reranking
        self.rerank_top_k = self.config.get("text_rerank_top_k", self.top_k * 2)
        # "cross-encoder/ms-marco-MiniLM-L6-v2"
        self.rerank_model_name = self.config.get(
            "text_rerank_model", "cross-encoder/ms-marco-MiniLM-L12-v2"
        )
        self.cross_encoder = None
        if self.use_rerank:
            try:
                rerank_device = "cuda" if torch.cuda.is_available() else "cpu"
                self.cross_encoder = CrossEncoder(self.rerank_model_name, device=rerank_device)
                logger.info(
                    f"Initialized CrossEncoder reranker {self.rerank_model_name} on {rerank_device}"
                )
            except Exception as e:
                logger.error(f"Failed to initialize CrossEncoder for reranking: {str(e)}")
                traceback.print_exc()
                self.use_rerank = False

        # Interactive text index state
        self._text_index_built = False
        self._bm25_model = None
        self._chroma_client = None
        self._text_collection = None
        self._text_chunks = []
        self._text_chunk_mapping = []

    def _rrf_fuse(self, rankings, k=60):
        fused_scores = {}
        for ranking in rankings:
            for idx, rank in ranking.items():
                fused_scores[idx] = fused_scores.get(idx, 0.0) + 1.0 / (k + rank)
        return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

    def _rerank_contexts(self, question, contexts):
        """Optionally rerank contexts using a CrossEncoder.

        Each element in contexts is expected to be a dict with at least a
        "chunk" field containing the text to be scored.
        """
        if not self.use_rerank or self.cross_encoder is None or not contexts:
            return contexts

        try:
            print(f"Reranking {len(contexts)} contexts for question: {question}")
            top_n = min(len(contexts), self.rerank_top_k)
            candidate_contexts = contexts[:top_n]
            pairs = [(question, ctx["chunk"]) for ctx in candidate_contexts]
            scores = self.cross_encoder.predict(pairs)

            for ctx, score in zip(candidate_contexts, scores):
                ctx["_rerank_score"] = float(score)

            candidate_contexts.sort(key=lambda x: x.get("_rerank_score", 0.0), reverse=True)
            return candidate_contexts
        except Exception as e:
            logger.error(f"Error during CrossEncoder reranking: {str(e)}")
            traceback.print_exc()
            return contexts

    def query_router_text(self, prompt: str) -> str:
        """Helper to generate a short text using the configured LLM.

        This is used for query transformation (rewrite, step-back, decomposition).
        """
        try:
            if self.llm_model == "gpt4":
                response = self.client.chat.completions.create(
                    model="gpt-4.1-nano",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=512,
                    temperature=0.0,
                )
                return response.choices[0].message.content.strip()

            if self.llm_model == "doubao":
                response = self.llm.responses.create(
                    model="doubao-seed-1-6-flash-250828",
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": prompt,
                                }
                            ],
                        }
                    ],
                )
                return response.output[1].content[0].text.strip()

            if self.llm_model == "qwen":
                messages = [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}],
                    }
                ]

                text = self.qwen_processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self.qwen_processor(
                    text=[text], padding=True, return_tensors="pt"
                ).to("cuda:0")

                generated_ids = self.qwen_model.generate(
                    **inputs, max_new_tokens=128
                )
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = self.qwen_processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True
                )
                return output_text[0].strip()

            logger.warning(
                f"Unsupported llm_model '{self.llm_model}' for query transformation; using original query."
            )
            return ""
        except Exception as e:
            logger.error(f"Error during LLM query transformation generation: {str(e)}")
            traceback.print_exc()
            return ""

    def _apply_query_transformation(self, original_query: str) -> str:
        """Apply configured query transformation before textual retrieval.

        Modes (via config["query_transform_mode"]):
        - "none" (default): return original_query unchanged
        - "rewrite": reformulate query to be more specific and detailed
        - "step_back": generate a broader, more general query
        - "decomposition": break query into sub-queries (used as a combined query string)
        """

        original_query = _normalize_chinese_numerals(original_query or "")
        mode = self.config.get("query_transform_mode", "none") or "none"
        mode = mode.lower()

        if mode == "none":
            return original_query

        try:
            if mode == "rewrite":
                prompt = (
                    "You are an AI assistant tasked with reformulating user queries to improve retrieval in a RAG system. Given the original query, rewrite it to be more specific, detailed, and likely to retrieve relevant information.\n\n"
                    f"Original query: {original_query}\n\nRewritten query:"
                )
            elif mode == "step_back":
                prompt = (
                    "You are an AI assistant tasked with generating broader, more general "
                    "queries to improve context retrieval in a RAG system. Given the "
                    "original query, generate a step-back query that is more general and "
                    "can help retrieve relevant background information.\n\n"
                    f"Original query: {original_query}\n\nStep-back query:"
                )
            elif mode == "decomposition":
                prompt = (
                    "You are an AI assistant tasked with breaking down complex queries into "
                    "2-4 simpler sub-queries for a RAG system. Given the original query, "
                    "decompose it into a short list of sub-queries that, when answered "
                    "together, would provide a comprehensive response. Write only the "
                    "sub-queries, each on its own line, without additional explanations.\n\n"
                    f"Original query: {original_query}\n\nSub-queries:"
                )
            else:
                logger.warning(
                    f"Unknown query_transform_mode '{mode}', using original query."
                )
                return original_query

            transformed = self.query_router_text(prompt).strip()
            if not transformed:
                return original_query

            # For decomposition we may receive multiple sub-queries; using the
            # entire transformed text as a single (richer) query string works
            # well with embedding/BM25 retrievers.
            return transformed
        except Exception as e:
            logger.error(
                f"Error applying query transformation (mode={mode}): {str(e)}"
            )
            traceback.print_exc()
            return original_query

    def build_text_index(self):
        """Build text index for all documents in the dataset."""
        logger.info(f"Building text index using {self.text_retriever}")

        try:
            # Cache documents if not already done
            if not self.parent.document_cache:
                self.parent.cache_documents()

            # Prepare all documents (each page is treated as a single chunk)
            all_chunks = []
            chunk_to_doc_mapping = []

            for doc_id, pages in self.parent.document_cache.items():
                for page_idx, page_text in enumerate(pages):
                    # Treat each page as a single chunk
                    chunk = _normalize_chinese_numerals(page_text or "")
                    all_chunks.append(chunk)
                    chunk_to_doc_mapping.append(
                        {
                            "chunk": chunk,
                            "chunk_pdf_name": doc_id,
                            "pdf_page_number": page_idx,
                        }
                    )

            # Initialize retriever based on selected method
            if self.text_retriever == "bm25":
                # BM25 indexing
                # bm25_model = BM25Okapi([chunk.split() for chunk in all_chunks])
                bm25_model = SimpleBM25(all_chunks)

                # Process each query
                results = []
                for _, row in self.df.iterrows():
                    q_id = row["q_id"]
                    question = _normalize_chinese_numerals(row["question"])

                    # Get BM25 scores
                    try:
                        scores = bm25_model.get_scores(question)
                        top_indices = sorted(
                            range(len(scores)),
                            key=lambda i: scores[i],
                            reverse=True,
                        )[: self.top_k * 2]

                        for rank, idx in enumerate(top_indices):
                            chunk_info = chunk_to_doc_mapping[idx]
                            results.append(
                                {
                                    "q_id": q_id,
                                    "question": question,
                                    "chunk": all_chunks[idx],
                                    "chunk_pdf_name": chunk_info["chunk_pdf_name"],
                                    "pdf_page_number": chunk_info["pdf_page_number"],
                                    "rank": rank + 1,
                                    "score": scores[idx],
                                }
                            )
                    except Exception as e:
                        logger.error(
                            f"Error processing query {q_id} with BM25: {str(e)}"
                        )
                        traceback.print_exc()

            elif self.text_retriever in ["minilm", "mpnet", "bge"]:
                # Dense-only retrieval using sentence transformer embeddings
                chroma_client = chromadb.Client()
                collection_name = f"st_col_{uuid.uuid4().hex[:8]}"
                collection = chroma_client.create_collection(
                    collection_name,
                    embedding_function=self.st_embedding_function,
                    metadata={"hnsw:space": "cosine"},
                )

                # Add documents to collection
                collection.add(
                    documents=all_chunks,
                    ids=[f"chunk_{i}" for i in range(len(all_chunks))],
                )

                # Process each query
                results = []
                for _, row in self.df.iterrows():
                    q_id = row["q_id"]
                    question = _normalize_chinese_numerals(row["question"])

                    # Get nearest chunks from dense retriever only
                    try:
                        query_results = collection.query(
                            query_texts=[question],
                            n_results=self.top_k * 2,
                        )

                        for rank, (chunk_idx, score) in enumerate(
                            zip(
                                [
                                    int(id.split("_")[1])
                                    for id in query_results["ids"][0]
                                ],
                                query_results["distances"][0],
                            )
                        ):
                            chunk_info = chunk_to_doc_mapping[chunk_idx]
                            results.append(
                                {
                                    "q_id": q_id,
                                    "question": question,
                                    "chunk": all_chunks[chunk_idx],
                                    "chunk_pdf_name": chunk_info["chunk_pdf_name"],
                                    "pdf_page_number": chunk_info["pdf_page_number"],
                                    "rank": rank + 1,
                                    # Convert distance to similarity
                                    "score": 1.0 - score,
                                }
                            )
                    except Exception as e:
                        logger.error(
                            f"Error processing query {q_id} with {self.text_retriever}: {str(e)}"
                        )
                        traceback.print_exc()

            elif self.text_retriever == "hybrid":
                # Hybrid retrieval: BM25 + dense (MiniLM) with RRF fusion
                # bm25_model = BM25Okapi([chunk.split() for chunk in all_chunks])
                bm25_model = SimpleBM25(all_chunks)

                chroma_client = chromadb.Client()
                collection_name = f"st_col_{uuid.uuid4().hex[:8]}"
                collection = chroma_client.create_collection(
                    collection_name,
                    embedding_function=self.st_embedding_function,
                    metadata={"hnsw:space": "cosine"},
                )

                collection.add(
                    documents=all_chunks,
                    ids=[f"chunk_{i}" for i in range(len(all_chunks))],
                )

                results = []
                for _, row in self.df.iterrows():
                    q_id = row["q_id"]
                    question = _normalize_chinese_numerals(row["question"])

                    try:
                        bm25_scores = bm25_model.get_scores(question)
                        bm25_top_indices = sorted(
                            range(len(bm25_scores)),
                            key=lambda i: bm25_scores[i],
                            reverse=True,
                        )[: self.top_k]
                        bm25_ranking = {
                            idx: rank + 1 for rank, idx in enumerate(bm25_top_indices)
                        }

                        query_results = collection.query(
                            query_texts=[question],
                            n_results=self.top_k,
                        )
                        dense_indices = [
                            int(id.split("_")[1]) for id in query_results["ids"][0]
                        ]
                        dense_ranking = {
                            idx: rank + 1 for rank, idx in enumerate(dense_indices)
                        }

                        fused = self._rrf_fuse([bm25_ranking, dense_ranking])
                        top_fused = fused[: self.top_k * 2]

                        for rank, (idx, fused_score) in enumerate(top_fused):
                            chunk_info = chunk_to_doc_mapping[idx]
                            results.append(
                                {
                                    "q_id": q_id,
                                    "question": question,
                                    "chunk": all_chunks[idx],
                                    "chunk_pdf_name": chunk_info["chunk_pdf_name"],
                                    "pdf_page_number": chunk_info["pdf_page_number"],
                                    "rank": rank + 1,
                                    "score": fused_score,
                                }
                            )
                    except Exception as e:
                        logger.error(
                            f"Error processing query {q_id} with {self.text_retriever}: {str(e)}"
                        )
                        traceback.print_exc()

            # Save results to CSV
            with open(self.text_retrieval_file, "w", newline="") as csvfile:
                fieldnames = [
                    "q_id",
                    "question",
                    "chunk",
                    "chunk_pdf_name",
                    "pdf_page_number",
                    "rank",
                    "score",
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)

            logger.info(f"Text index saved to {self.text_retrieval_file}")
            return True

        except Exception as e:
            logger.error(f"Error building text index: {str(e)}")
            traceback.print_exc()
            return False

    def retrieve_textual_contexts(self, query_id):
        """Retrieve textual contexts for a benchmark query by ID."""
        try:
            if not os.path.exists(self.text_retrieval_file) or self.force_reindex:
                logger.info(
                    "Textual index not found or force reindex is enabled. Building index..."
                )
                if not self.build_text_index():
                    return []

            retrieval_path = self.text_retrieval_file

            df_retrieval = pd.read_csv(retrieval_path)

            # Filter for the current query
            query_rows = df_retrieval[df_retrieval["q_id"] == query_id]
            if len(query_rows) == 0:
                logger.warning(f"No textual contexts found for query {query_id}")
                return []

            # Get top-k textual contexts
            top_k_rows = query_rows.sort_values(by="rank", ascending=True).head(
                self.top_k
            )

            # Extract the contexts
            contexts = []
            for _, row in top_k_rows.iterrows():
                contexts.append(
                    {
                        "chunk": row["chunk"],
                        "chunk_pdf_name": row["chunk_pdf_name"]
                        if "chunk_pdf_name" in row
                        else row.get("document_id", "unknown"),
                        "pdf_page_number": row["pdf_page_number"]
                        if "pdf_page_number" in row
                        else row.get("page_number", 0),
                    }
                )

            logger.info(f"Retrieved {len(contexts)} textual contexts for query {query_id}")
            return contexts

        except Exception as e:
            logger.error(f"Error retrieving textual contexts for query {query_id}: {str(e)}")
            traceback.print_exc()
            return []

    @dataclass
    class ChunkInfo:
        """分块信息"""
        content: str
        doc_id: str
        page_idx: int
        chunk_type: str  # 'text', 'table', 'image'
        headers: Dict[str, str] = None  # Markdown标题层级
        table_summary: str = None  # 表格摘要

    def _preprocess_text(self, text: str) -> str:
        """预处理OCR文本"""
        if not text:
            return ""
        
        text = _normalize_chinese_numerals(text)
        
        # 清理多余空白
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r' +\n', '\n', text)
        
        # 修复Markdown标题格式
        text = re.sub(r'^(#+)([^\s#])', r'\1 \2', text, flags=re.MULTILINE)
        
        return text.strip()

    def _extract_table_caption(self, text_after_table: str):
        if not text_after_table:
            return "", text_after_table or ""
        match = re.search(
            r"^\s*caption:(.*)$",
            text_after_table,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        if not match:
            return "", text_after_table
        caption = match.group(1).strip()
        start, end = match.span()
        remaining = (text_after_table[:start] + text_after_table[end:]).lstrip("\n")
        return caption, remaining

    def _handle_table_page(
        self,
        doc_id,
        page_idx,
        text,
        all_chunks,
        chunk_to_doc_mapping,
        html_table_pattern,
        split_markdown_segment,
    ):
        last_pos = 0
        table_matches = list(html_table_pattern.finditer(text))

        for i, match in enumerate(table_matches):
            start, end = match.span()

            parts = []

            # Pure text before this table (since last table or page start)
            if start > last_pos:
                prefix = text[last_pos:start]
                if prefix.strip():
                    parts.append(prefix)

            # Text between the end of this table and the next table (or page end)
            next_start = (
                table_matches[i + 1].start()
                if i + 1 < len(table_matches)
                else len(text)
            )
            post_table_full = text[end:next_start]
            caption, post_table = self._extract_table_caption(post_table_full)

            if caption:
                parts.append(caption)
            if post_table and post_table.strip():
                parts.append(post_table)

            if parts:
                region_text = "\n".join(seg.strip() for seg in parts)
                region_chunks = split_markdown_segment(region_text)

                if caption:
                    caption_chunks = [
                        idx
                        for idx, chunk in enumerate(region_chunks)
                        if caption in chunk
                    ]
                    if len(caption_chunks) > 1:
                        first = caption_chunks[0]
                        last_idx = caption_chunks[-1]
                        merged = "\n".join(region_chunks[first : last_idx + 1])
                        region_chunks = (
                            region_chunks[:first]
                            + [merged]
                            + region_chunks[last_idx + 1 :]
                        )

                for chunk in region_chunks:
                    all_chunks.append(chunk)
                    chunk_to_doc_mapping.append(
                        {
                            "chunk": chunk,
                            "chunk_pdf_name": doc_id,
                            "pdf_page_number": page_idx,
                        }
                    )

            # Next table can only see text after this one
            last_pos = next_start

    def _handle_image_page(
        self,
        doc_id,
        page_idx,
        text,
        all_chunks,
        chunk_to_doc_mapping,
        image_block_pattern,
        split_markdown_segment,
    ):
        last_idx = 0
        text_segments = []
        image_blocks = []

        for match in image_block_pattern.finditer(text):
            # Text segment before this image
            if match.start() > last_idx:
                prefix = text[last_idx:match.start()]
                if prefix.strip():
                    text_segments.append(prefix)

            # The image block itself (URL + optional caption)
            block = text[match.start():match.end()]
            if block.strip():
                image_blocks.append(block.strip())

            last_idx = match.end()

        # Trailing text after the last image
        if last_idx < len(text):
            suffix = text[last_idx:]
            if suffix.strip():
                text_segments.append(suffix)

        # Merge all pure-text segments on the page into a single logical stream
        # and then chunk it. This way, text before and after images can end up
        # in the same chunk if the page is not too long.
        if text_segments:
            merged_text = "\n".join(seg.strip() for seg in text_segments)
            for chunk in split_markdown_segment(merged_text):
                all_chunks.append(chunk)
                chunk_to_doc_mapping.append(
                    {
                        "chunk": chunk,
                        "chunk_pdf_name": doc_id,
                        "pdf_page_number": page_idx,
                    }
                )

        # Add image blocks as independent chunks so their URL+caption remain atomic
        for block in image_blocks:
            all_chunks.append(block)
            chunk_to_doc_mapping.append(
                {
                    "chunk": block,
                    "chunk_pdf_name": doc_id,
                    "pdf_page_number": page_idx,
                }
            )

    def _handle_plain_text_page(
        self,
        doc_id,
        page_idx,
        text,
        all_chunks,
        chunk_to_doc_mapping,
        split_markdown_segment,
    ):
        for chunk in split_markdown_segment(text):
            all_chunks.append(chunk)
            chunk_to_doc_mapping.append(
                {
                    "chunk": chunk,
                    "chunk_pdf_name": doc_id,
                    "pdf_page_number": page_idx,
                }
            )

    def _build_interactive_text_index(self):
        if self._text_index_built:
            return
        if not self.parent.document_cache:
            self.parent.cache_documents()
        all_chunks = []
        chunk_to_doc_mapping = []
        # Build one chunk per document page to align textual retrieval with page-level granularity

         # 改进的分隔符列表
        self.separators = [
            "\n\n",                          # 段落
            "\n",                            # 换行
            "。", "！", "？",          # 中文标点
            ".\n", ". ", "! ", "? ",  # 英文标点+空格（避免切断小数点）
            " ",                             # 空格
            ""                               # 字符级
        ]
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.get("text_chunk_size", 500),
            chunk_overlap=self.config.get("text_chunk_overlap", 100),
            separators=self.separators,
            keep_separator=True   # 保持分隔符
        )
        

        md_header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ],
            strip_headers=False,
        )

        def _split_markdown_segment(segment: str):
            if not segment or not segment.strip():
                return []
            try:
                docs = md_header_splitter.split_text(segment)
                if not docs:
                    return splitter.split_text(segment)
                split_docs = splitter.split_documents(docs)
                return [d.page_content for d in split_docs]
            except Exception:
                return splitter.split_text(segment)

        # Tables are rendered in markdown as HTML <table>...</table> blocks.
        # A single page may contain multiple tables, but any presence of a
        # <table> block means we treat the whole page as one chunk.
        html_table_pattern = re.compile(r"<table\b.*?</table>", re.IGNORECASE | re.DOTALL)

        image_block_pattern = re.compile(
            r"!\[[^\]]*\]\([^)]+\)(?:\s*\n\s*caption:.*)?",
            re.IGNORECASE,
        )
        for doc_id, pages in tqdm(self.parent.document_cache.items(),desc="Splitting differenct type texts into chunks..."):
            for page_idx, page_text in enumerate(pages):
                text = self._preprocess_text(page_text or "")
                # If the page contains HTML table blocks, let the table handler split
                # the page into table chunks and surrounding pure-text chunks.
                if html_table_pattern.search(text):
                    self._handle_table_page(
                        doc_id,
                        page_idx,
                        text,
                        all_chunks,
                        chunk_to_doc_mapping,
                        html_table_pattern,
                        _split_markdown_segment,
                    )
                    continue

                # If the page contains images (with optional captions), keep image path and
                # caption together as atomic blocks while chunking the surrounding text.
                if image_block_pattern.search(text):
                    self._handle_image_page(
                        doc_id,
                        page_idx,
                        text,
                        all_chunks,
                        chunk_to_doc_mapping,
                        image_block_pattern,
                        _split_markdown_segment,
                    )
                else:
                    self._handle_plain_text_page(
                        doc_id,
                        page_idx,
                        text,
                        all_chunks,
                        chunk_to_doc_mapping,
                        _split_markdown_segment,
                    )

        self._text_chunks = all_chunks
        self._text_chunk_mapping = chunk_to_doc_mapping
        logger.info(f"Built interactive text index with {len(all_chunks)} chunks")
        # BM25 index is needed for bm25 and hybrid modes
        if self.text_retriever in ["bm25", "minilm", "mpnet", "bge", "hybrid"]:
            if self.text_retriever in ["bm25", "hybrid"]:
                # self._bm25_model = BM25Okapi([chunk.split() for chunk in all_chunks])
                self._bm25_model = SimpleBM25(all_chunks)
            # Dense index is needed for dense-only and hybrid modes
            if self.text_retriever in ["minilm", "mpnet", "bge", "hybrid"]:
                chroma_client = chromadb.Client()
                collection_name = f"st_col_{uuid.uuid4().hex[:8]}"
                collection = chroma_client.create_collection(
                    collection_name,
                    embedding_function=self.st_embedding_function,
                    metadata={"hnsw:space": "cosine"},
                )
                collection.add(
                    documents=all_chunks,
                    ids=[f"chunk_{i}" for i in range(len(all_chunks))],
                )
                self._chroma_client = chroma_client
                self._text_collection = collection
        else:
            raise ValueError(f"Unsupported text retriever: {self.text_retriever}")
        self._text_index_built = True

    def retrieve_textual_contexts_for_question(self, question):
        try:
            self._build_interactive_text_index()
            if not self._text_index_built:
                return []
            transformed_question = self._apply_query_transformation(question)
            retrieval_query = transformed_question or _normalize_chinese_numerals(
                question or ""
            )
            contexts = []
            if self.text_retriever == "bm25":
                scores = self._bm25_model.get_scores(retrieval_query)
                pre_k = self.rerank_top_k if self.use_rerank else self.top_k
                top_indices = sorted(
                    range(len(scores)), key=lambda i: scores[i], reverse=True
                )[: pre_k]
                for idx in top_indices:
                    chunk_info = self._text_chunk_mapping[idx]
                    contexts.append(
                        {
                            "chunk": self._text_chunks[idx],
                            "chunk_pdf_name": chunk_info["chunk_pdf_name"],
                            "pdf_page_number": chunk_info["pdf_page_number"],
                        }
                    )
            elif self.text_retriever in ["minilm", "mpnet", "bge"]:
                # Dense-only interactive retrieval
                pre_k = self.rerank_top_k if self.use_rerank else self.top_k
                query_results = self._text_collection.query(
                    query_texts=[retrieval_query],
                    n_results=pre_k,
                )
                for chunk_idx, score in zip(
                    [int(id.split("_")[1]) for id in query_results["ids"][0]],
                    query_results["distances"][0],
                ):
                    chunk_info = self._text_chunk_mapping[chunk_idx]
                    contexts.append(
                        {
                            "chunk": self._text_chunks[chunk_idx],
                            "chunk_pdf_name": chunk_info["chunk_pdf_name"],
                            "pdf_page_number": chunk_info["pdf_page_number"],
                        }
                    )
            elif self.text_retriever == "hybrid":
                # Hybrid interactive retrieval: BM25 + dense with RRF fusion
                bm25_scores = self._bm25_model.get_scores(retrieval_query)
                pre_k = self.rerank_top_k if self.use_rerank else self.top_k * 2

                bm25_top_indices = sorted(
                    range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
                )[: pre_k]
                bm25_ranking = {
                    idx: rank + 1 for rank, idx in enumerate(bm25_top_indices)
                }

                query_results = self._text_collection.query(
                    query_texts=[retrieval_query],
                    n_results=pre_k,
                )
                dense_indices = [
                    int(id.split("_")[1]) for id in query_results["ids"][0]
                ]
                dense_ranking = {
                    idx: rank + 1 for rank, idx in enumerate(dense_indices)
                }

                fused = self._rrf_fuse([bm25_ranking, dense_ranking])
                top_indices = [idx for idx, _ in fused[: pre_k]]
                for idx in top_indices:
                    chunk_info = self._text_chunk_mapping[idx]
                    contexts.append(
                        {
                            "chunk": self._text_chunks[idx],
                            "chunk_pdf_name": chunk_info["chunk_pdf_name"],
                            "pdf_page_number": chunk_info["pdf_page_number"],
                        }
                    )
            else:
                return []

            # Optional CrossEncoder reranking of interactive contexts
            contexts = self._rerank_contexts(question, contexts)
            if len(contexts) > self.top_k:
                contexts = contexts[: self.top_k]
            logger.info(
                "Retrieved %d textual contexts for interactive query", len(contexts)
            )
            return contexts
        except Exception as e:
            logger.error(
                "Error retrieving textual contexts for interactive query: %s", str(e)
            )
            traceback.print_exc()
            return []

    def generate_textual_response(self, query, textual_contexts):
        """Generate a response based on textual contexts."""
        try:
            # Extract the text chunks
            contexts = [ctx["chunk"] for ctx in textual_contexts]
            contexts_str = "\n- ".join(contexts)

            # Create prompt
            prompt_template = f"""
            You are tasked with answering a question based on the relevant chunks of a PDF document. Provide your response in the following format:
            ## Evidence:

            ## Chain of Thought:

            ## Answer:

            ___
            Instructions:

            1. Evidence Curation: Extract relevant elements (such as paragraphs, tables, figures, charts) from the provided chunks and populate them in the "Evidence" section. For each element, include the type, content, and a brief explanation of its relevance.

            2. Chain of Thought: In the "Chain of Thought" section, list out each logical step you take to derive the answer, referencing the evidence where applicable. You should perform computations if you need to to get to the answer. 

            3. Answer: {self.qa_prompt}
            ___
            Question: {query}
            ___
            Context: {contexts_str}
            
            """

            if self.llm_model == "gpt4":
                response = self.client.chat.completions.create(
                    model="chatgpt-4o-latest",
                    messages=[{"role": "user", "content": prompt_template}],
                    max_tokens=3000,
                    temperature=0.7,
                )
                return response.choices[0].message.content

            elif self.llm_model == "doubao":
                # Non-streaming:
                print("----- standard request -----")
                # completion = self.llm.chat.completions.create(
                #     model="doubao-1-5-lite-32k-250115",
                #     messages=[
                #         {"role": "user", "content": prompt_template},
                #     ],
                # )
                # return completion.choices[0].message.content
                response = self.llm.responses.create(
                    model="doubao-seed-1-6-flash-250828",
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": prompt_template,
                                }
                            ],
                        }
                    ],
                )
                return response.output[1].content[0].text

            elif self.llm_model == "qwen":
                messages = [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt_template}],
                    }
                ]

                text = self.qwen_processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self.qwen_processor(
                    text=[text], padding=True, return_tensors="pt"
                ).to("cuda:0")

                generated_ids = self.qwen_model.generate(**inputs, max_new_tokens=512)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = self.qwen_processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True
                )

                return output_text[0]

        except Exception as e:
            logger.error(f"Error generating textual response: {str(e)}")
            return "Error generating response from textual contexts."


__all__ = ["TextualRAGEngine"]
