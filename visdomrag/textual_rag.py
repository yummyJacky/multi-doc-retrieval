import logging
import os
import traceback
import csv
import uuid

import chromadb
import pandas as pd
import torch
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi


logger = logging.getLogger("VisDoMRAG")


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
            self.st_embedding_function = None
        elif self.text_retriever in ["minilm", "mpnet", "bge"]:
            # Map text_retriever to actual model names
            model_map = {
                "minilm": "sentence-transformers/all-MiniLM-L6-v2",
                "mpnet": "sentence-transformers/all-mpnet-base-v2",
                "bge": "BAAI/bge-base-en-v1.5",
            }

            # Load sentence transformer model
            self.text_model_name = model_map[self.text_retriever]
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.st_embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.text_model_name, device=self.device
            )
        else:
            raise ValueError(f"Unsupported text retriever: {self.text_retriever}")

        # Interactive text index state
        self._text_index_built = False
        self._bm25_model = None
        self._chroma_client = None
        self._text_collection = None
        self._text_chunks = []
        self._text_chunk_mapping = []

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
                    chunk = page_text
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
                bm25_model = BM25Okapi([chunk.split() for chunk in all_chunks])

                # Process each query
                results = []
                for _, row in self.df.iterrows():
                    q_id = row["q_id"]
                    question = row["question"]

                    # Get BM25 scores
                    try:
                        scores = bm25_model.get_scores(question.split())
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
                # Create Chroma collection with sentence transformer embeddings
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
                    question = row["question"]

                    # Get nearest chunks
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
                                    "score": 1.0 - score,  # Convert distance to similarity
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

    def _build_interactive_text_index(self):
        if self._text_index_built:
            return
        if not self.parent.document_cache:
            self.parent.cache_documents()
        all_chunks = []
        chunk_to_doc_mapping = []
        # Build one chunk per document page to align textual retrieval with page-level granularity
        for doc_id, pages in self.parent.document_cache.items():
            for page_idx, page_text in enumerate(pages):
                chunk = page_text
                all_chunks.append(chunk)
                chunk_to_doc_mapping.append(
                    {
                        "chunk": chunk,
                        "chunk_pdf_name": doc_id,
                        # Keep page index 0-based to align with visual index page indices
                        "pdf_page_number": page_idx,
                    }
                )

        self._text_chunks = all_chunks
        self._text_chunk_mapping = chunk_to_doc_mapping
        logger.info(f"Built interactive text index with {len(all_chunks)} chunks")
        if self.text_retriever == "bm25":
            self._bm25_model = BM25Okapi([chunk.split() for chunk in all_chunks])
        elif self.text_retriever in ["minilm", "mpnet", "bge"]:
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
            contexts = []
            if self.text_retriever == "bm25":
                scores = self._bm25_model.get_scores(question.split())
                top_indices = sorted(
                    range(len(scores)), key=lambda i: scores[i], reverse=True
                )[: self.top_k]
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
                query_results = self._text_collection.query(
                    query_texts=[question],
                    n_results=self.top_k,
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
            else:
                return []
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
