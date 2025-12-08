import os
import json
import torch
import pandas as pd
import numpy as np
import time
import logging
import argparse
import re
import uuid
import csv
from tqdm import tqdm
from io import BytesIO
from pdf2image import convert_from_path
import base64
import requests
from PIL import Image
import gc
from difflib import SequenceMatcher
import PyPDF2
import pytesseract
import traceback
from openai import OpenAI
from dots_ocr.parser import DotsOCRParser
from transformers import AutoModel

# For embeddings and retrieval
try:
    import chromadb
    import chromadb.utils.embedding_functions as embedding_functions
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from rank_bm25 import BM25Okapi
except ImportError as e:
    print(f"Optional dependency missing: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("visdmrag.log"), logging.StreamHandler()]
)
logger = logging.getLogger("VisDoMRAG")


class VisualRAGEngine:
    def __init__(self, parent):
        """Engine responsible for visual retrieval and visual QA."""
        self.parent = parent
        self.config = parent.config
        self.data_dir = parent.data_dir
        self.output_dir = parent.output_dir
        self.llm_model = parent.llm_model
        self.vision_retriever = parent.vision_retriever
        self.top_k = parent.top_k
        self.force_reindex = parent.force_reindex
        self.qa_prompt = parent.qa_prompt
        self.pdf_files = parent.pdf_files
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
        if hasattr(parent, "process_vision_info"):
            self.process_vision_info = parent.process_vision_info

        # Visual retrieval resources (supporting ColPali/ColQwen and Nvidia NemoRetriever)
        self.vision_retrieval_file = (
            f"{self.data_dir}/retrieval/retrieval_{self.vision_retriever}.csv"
        )

        # Model / processor handles
        self.vision_model = None
        self.vision_processor = None

        # Keep original ColPali / ColQwen flows, and add Nemo as an alternative
        if self.vision_retriever in ["colpali", "colqwen"]:
            try:
                if self.vision_retriever == "colpali":
                    from colpali_engine.models import ColPali, ColPaliProcessor

                    logger.info("Loading ColPali model for visual indexing")
                    self.vision_model = ColPali.from_pretrained(
                        "vidore/colpali-v1.2",
                        torch_dtype=torch.bfloat16,
                        device_map="cuda",
                    ).eval()
                    self.vision_processor = ColPaliProcessor.from_pretrained(
                        "vidore/colpali-v1.2"
                    )
                else:  # colqwen
                    from colpali_engine.models import ColQwen2, ColQwen2Processor

                    logger.info("Loading ColQwen model for visual indexing")
                    self.vision_model = ColQwen2.from_pretrained(
                        "vidore/colqwen2-v1.0",
                        torch_dtype=torch.bfloat16,
                        device_map="cuda",
                    ).eval()
                    self.vision_processor = ColQwen2Processor.from_pretrained(
                        "vidore/colqwen2-v0.1"
                    )
            except ImportError:
                raise ImportError(
                    "ColPali/ColQwen models not found. Please install colpali_engine."
                )
        elif self.vision_retriever in ["nemo", "nvidia", "nemo_retriever"]:
            try:
                logger.info(
                    "Loading Nvidia NemoRetriever model 'nvidia/llama-nemoretriever-colembed-1b-v1' for visual indexing"
                )
                self.vision_model = AutoModel.from_pretrained(
                    "nvidia/llama-nemoretriever-colembed-1b-v1",
                    device_map="cuda",
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                    revision="1f0fdea7f5b19532a750be109b19072d719b8177",
                ).eval()
            except ImportError as e:
                raise ImportError(
                    f"Failed to import transformers for Nvidia NemoRetriever: {e}. Please install 'transformers'."
                )
            except Exception as e:
                logger.error("Error loading Nvidia NemoRetriever model: %s", str(e))
                raise
        else:
            raise ValueError(f"Unsupported visual retriever: {self.vision_retriever}")

        # Interactive visual index state
        self._visual_index_built = False
        self._page_embeddings = None  # Tensor of shape [n_pages, n_tokens, dim]
        self._page_ids = []
        self._page_info = {}

    def build_visual_index(self):
        """Build visual embedding index for all PDFs in the dataset using multi-vector scoring."""
        logger.info(f"Building visual index using {self.vision_retriever}")

        try:
            pdf_dir = os.path.join(self.data_dir, "docs")
            output_dir = os.path.join(self.data_dir, "visual_embeddings")
            os.makedirs(output_dir, exist_ok=True)

            # Extract unique document IDs from the dataset
            unique_docs = set()
            for _, row in self.df.iterrows():
                try:
                    docs = eval(row["documents"]) if "documents" in row else []
                    unique_docs.update(docs)
                except Exception:
                    traceback.print_exc()
                    pass

                if "doc_path" in row:
                    doc_path = row["doc_path"]
                    if isinstance(doc_path, str) and doc_path.strip():
                        unique_docs.add(os.path.basename(doc_path))

            # Get list of PDF files based on unique_docs
            pdf_files = []
            for doc_id in unique_docs:
                if doc_id.endswith(".pdf"):
                    pdf_files.append(doc_id)
                else:
                    pdf_files.append(f"{doc_id}.pdf")

            results = []

            # --- Branch 1: original ColPali / ColQwen multi-vector flow ---
            if self.vision_retriever in ["colpali", "colqwen"]:
                if self.vision_processor is None or self.vision_model is None:
                    raise RuntimeError(
                        "ColPali/ColQwen vision model or processor is not initialized."
                    )

                page_embeddings = {}

                # Process each PDF
                for pdf_file in tqdm(pdf_files, desc="Processing PDFs for visual index"):
                    doc_id = os.path.splitext(pdf_file)[0]
                    pdf_path = os.path.join(pdf_dir, pdf_file)

                    if not os.path.exists(pdf_path):
                        logger.warning(f"PDF file not found: {pdf_path}")
                        continue

                    try:
                        pages = convert_from_path(pdf_path)
                    except Exception as e:
                        logger.error(
                            f"Error converting PDF {pdf_file} to images: {str(e)}"
                        )
                        traceback.print_exc()
                        continue

                    for page_idx, page_img in enumerate(pages):
                        page_id = f"{doc_id}_{page_idx}"
                        try:
                            processed_image = self.vision_processor.process_images([page_img])
                            processed_image = {
                                k: v.to(self.vision_model.device)
                                for k, v in processed_image.items()
                            }
                            with torch.no_grad():
                                embedding = self.vision_model(**processed_image)

                            # Save embedding to file (keep original behavior)
                            embedding_file = os.path.join(
                                output_dir, f"{page_id}.pt"
                            )
                            torch.save(embedding.cpu(), embedding_file)

                            # Cache for multi-vector scoring
                            page_embeddings[page_id] = embedding.cpu()
                        except Exception as e:
                            logger.error(
                                f"Error processing page {page_idx} of PDF {pdf_file}: {str(e)}"
                            )
                            traceback.print_exc()
                            continue

                # Generate query embeddings
                query_embeddings = {}
                for _, row in tqdm(
                    self.df.iterrows(), desc="Processing queries for visual index"
                ):
                    q_id = row["q_id"]
                    question = row["question"]

                    try:
                        processed_query = self.vision_processor.process_queries([question])
                        processed_query = {
                            k: v.to(self.vision_model.device)
                            for k, v in processed_query.items()
                        }
                        with torch.no_grad():
                            embedding = self.vision_model(**processed_query)

                        query_embeddings[q_id] = embedding.cpu()

                        # Save query embedding
                        query_embedding_file = os.path.join(
                            output_dir, f"query_{q_id}.pt"
                        )
                        torch.save(embedding.cpu(), query_embedding_file)
                    except Exception as e:
                        logger.error(
                            f"Error generating embedding for query {q_id}: {str(e)}"
                        )
                        traceback.print_exc()

                # Multi-vector scoring per query
                for _, row in tqdm(
                    self.df.iterrows(), desc="Ranking documents for queries"
                ):
                    q_id = row["q_id"]
                    question = row["question"]

                    query_emb = query_embeddings.get(q_id)
                    if query_emb is None:
                        continue

                    relevant_docs = []
                    if "documents" in row:
                        try:
                            docs = eval(row["documents"])
                            relevant_docs = [doc.split(".pdf")[0] for doc in docs]
                        except Exception:
                            traceback.print_exc()
                            relevant_docs = [
                                os.path.splitext(f)[0] for f in pdf_files
                            ]

                    if not relevant_docs:
                        relevant_docs = [os.path.splitext(f)[0] for f in pdf_files]

                    relevant_page_embeddings = {}
                    for page_id, embedding in page_embeddings.items():
                        doc_id = page_id.rsplit("_", 1)[0]
                        if doc_id in relevant_docs:
                            relevant_page_embeddings[page_id] = embedding

                    if not relevant_page_embeddings:
                        logger.warning(
                            f"No relevant document embeddings found for query {q_id}"
                        )
                        continue

                    qs = query_emb
                    ds = torch.cat(
                        [emb for emb in relevant_page_embeddings.values()], dim=0
                    )

                    try:
                        scores = self.vision_processor.score_multi_vector(qs, ds)
                        scores = scores.flatten().numpy()
                    except Exception as e:
                        logger.error(
                            f"Error scoring documents for query {q_id}: {str(e)}"
                        )
                        traceback.print_exc()
                        continue

                    top_indices = np.argsort(-scores)
                    ranked_docs = np.array(
                        list(relevant_page_embeddings.keys())
                    )[top_indices]

                    for doc_id, score in zip(ranked_docs, scores[top_indices]):
                        results.append(
                            {
                                "q_id": q_id,
                                "document_id": doc_id,
                                "score": float(score),
                                "question": question,
                            }
                        )

            # --- Branch 2: Nvidia NemoRetriever multi-vector flow ---
            elif self.vision_retriever in ["nemo", "nvidia", "nemo_retriever"]:
                # Track all generated embeddings
                page_ids = []
                page_embeddings_list = []  # list of [n_tokens, dim]

                # Process each PDF
                for pdf_file in tqdm(pdf_files, desc="Processing PDFs for visual index"):
                    doc_id = os.path.splitext(pdf_file)[0]
                    pdf_path = os.path.join(pdf_dir, pdf_file)

                    # Skip if file doesn't exist
                    if not os.path.exists(pdf_path):
                        logger.warning(f"PDF file not found: {pdf_path}")
                        continue

                    # Convert PDF to images
                    try:
                        pages = convert_from_path(pdf_path)
                    except Exception as e:
                        logger.error(
                            f"Error converting PDF {pdf_file} to images: {str(e)}"
                        )
                        traceback.print_exc()
                        continue

                    if not pages:
                        continue

                    # Encode all pages of this PDF with NemoRetriever
                    try:
                        with torch.no_grad():
                            passage_embeddings = self.vision_model.forward_passages(
                                pages, batch_size=8
                            )
                        passage_embeddings = passage_embeddings.cpu()
                    except Exception as e:
                        logger.error(
                            f"Error encoding pages for PDF {pdf_file} with NemoRetriever: {str(e)}"
                        )
                        traceback.print_exc()
                        continue

                    # Store per-page embeddings
                    for page_idx in range(len(pages)):
                        page_id = f"{doc_id}_{page_idx}"
                        page_ids.append(page_id)
                        page_embeddings_list.append(passage_embeddings[page_idx])

                if not page_embeddings_list:
                    logger.warning(
                        "No page embeddings were generated for visual index."
                    )
                    return False

                # Stack to a single tensor [n_pages, n_tokens, dim]
                page_embeddings = torch.stack(page_embeddings_list, dim=0)

                # Generate query embeddings
                query_embeddings = {}
                for _, row in tqdm(
                    self.df.iterrows(), desc="Processing queries for visual index"
                ):
                    q_id = row["q_id"]
                    question = row["question"]

                    try:
                        with torch.no_grad():
                            q_emb = self.vision_model.forward_queries(
                                [question], batch_size=1
                            )
                        query_embeddings[q_id] = q_emb.cpu()
                    except Exception as e:
                        logger.error(
                            f"Error generating embedding for query {q_id}: {str(e)}"
                        )
                        traceback.print_exc()

                # Use NemoRetriever get_scores to rank documents for each query
                for _, row in tqdm(
                    self.df.iterrows(), desc="Ranking documents for queries"
                ):
                    q_id = row["q_id"]
                    question = row["question"]

                    query_emb = query_embeddings.get(q_id)
                    if query_emb is None:
                        continue

                    # Get relevant documents for this query based on the dataset
                    relevant_docs = []
                    if "documents" in row:
                        try:
                            docs = eval(row["documents"])
                            relevant_docs = [doc.split(".pdf")[0] for doc in docs]
                        except Exception:
                            # If documents field is not valid, use all documents
                            traceback.print_exc()
                            relevant_docs = [
                                os.path.splitext(f)[0] for f in pdf_files
                            ]

                    if not relevant_docs:
                        relevant_docs = [os.path.splitext(f)[0] for f in pdf_files]

                    # Select candidate pages
                    candidate_indices = []
                    candidate_page_ids = []
                    for idx, pid in enumerate(page_ids):
                        doc_id = pid.rsplit("_", 1)[0]
                        if doc_id in relevant_docs:
                            candidate_indices.append(idx)
                            candidate_page_ids.append(pid)

                    if not candidate_indices:
                        logger.warning(
                            f"No relevant document embeddings found for query {q_id}"
                        )
                        continue

                    candidate_embeddings = page_embeddings[candidate_indices]

                    try:
                        with torch.no_grad():
                            scores = self.vision_model.get_scores(
                                query_emb, candidate_embeddings
                            )
                        query_scores = scores[0].cpu().numpy()
                    except Exception as e:
                        logger.error(
                            f"Error ranking documents for query {q_id}: {str(e)}"
                        )
                        traceback.print_exc()
                        continue

                    # Get indices of scores in descending order
                    top_indices = np.argsort(-query_scores)

                    # Store results for each ranked document
                    for idx in top_indices:
                        doc_page_id = candidate_page_ids[idx]
                        score = float(query_scores[idx])
                        results.append(
                            {
                                "q_id": q_id,
                                "document_id": doc_page_id,
                                "score": score,
                                "question": question,
                            }
                        )
            else:
                raise ValueError(f"Unsupported visual retriever: {self.vision_retriever}")

            # Save results to CSV
            with open(self.vision_retrieval_file, "w", newline="") as csvfile:
                fieldnames = ["q_id", "document_id", "score", "question"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)

            logger.info(f"Visual index saved to {self.vision_retrieval_file}")
            return True

        except Exception as e:
            logger.error(f"Error building visual index: {str(e)}")
            traceback.print_exc()
            return False

    def retrieve_visual_contexts(self, query_id):
        """Retrieve visual contexts (images) for a benchmark query by ID."""
        try:
            # Check if we need to build the index
            if not os.path.exists(self.vision_retrieval_file) or self.force_reindex:
                logger.info(
                    "Visual index not found or force reindex is enabled. Building index..."
                )
                if not self.build_visual_index():
                    return []

            # Load the retrieval results
            df_retrieval = pd.read_csv(self.vision_retrieval_file)

            # Filter for the current query
            query_rows = df_retrieval[df_retrieval["q_id"] == query_id]
            if len(query_rows) == 0:
                logger.warning(f"No visual contexts found for query {query_id}")
                return []

            # Get top-k visual contexts
            top_k_rows = query_rows.nlargest(self.top_k, "score")

            # Load the images from PDFs
            pages = []
            pdf_dir = os.path.join(self.data_dir, "docs")

            for _, row in top_k_rows.iterrows():
                try:
                    document_id = row["document_id"]
                    # Extract base document ID and page number
                    base_doc_id, page_number = document_id.rsplit("_", 1)
                    page_number = int(page_number)

                    # Find the PDF file
                    pdf_path = os.path.join(pdf_dir, f"{base_doc_id}.pdf")
                    if not os.path.exists(pdf_path):
                        logger.warning(f"PDF file not found: {pdf_path}")
                        continue

                    # Convert PDF page to image
                    pdf_images = convert_from_path(pdf_path)
                    if page_number >= len(pdf_images):
                        logger.warning(
                            f"Page {page_number} out of range for {pdf_path}"
                        )
                        continue

                    image = pdf_images[page_number]
                    pages.append(
                        {
                            "image": image,
                            "document_id": document_id,
                            "page_number": page_number,
                        }
                    )
                except Exception as e:
                    logger.error(
                        f"Error loading PDF page for {row['document_id']}: {str(e)}"
                    )
                    traceback.print_exc()

            logger.info(f"Retrieved {len(pages)} visual contexts for query {query_id}")
            return pages

        except Exception as e:
            logger.error(f"Error retrieving visual contexts for query {query_id}: {str(e)}")
            traceback.print_exc()
            return []

    def _build_interactive_visual_index(self):
        if self._visual_index_built:
            return
        if not self.pdf_files:
            logger.warning("pdf_files not provided for interactive visual index")
            self._visual_index_built = False
            return
        pdf_paths = self.parent._get_config_pdf_paths()
        if not pdf_paths:
            logger.warning("No valid PDF files for interactive visual index")
            self._visual_index_built = False
            return
        # ColPali / ColQwen interactive index
        if self.vision_retriever in ["colpali", "colqwen"]:
            page_embeddings = {}
            page_info = {}
            for doc_id, pdf_path in tqdm(
                pdf_paths.items(), desc="Processing PDFs for interactive visual index"
            ):
                try:
                    pages = convert_from_path(pdf_path)
                except Exception as e:
                    logger.error(
                        f"Error converting PDF {pdf_path} to images: {str(e)}"
                    )
                    traceback.print_exc()
                    continue
                for page_idx, page_img in enumerate(pages):
                    page_id = f"{doc_id}_{page_idx}"
                    try:
                        processed_image = self.vision_processor.process_images([page_img])
                        processed_image = {
                            k: v.to(self.vision_model.device)
                            for k, v in processed_image.items()
                        }
                        with torch.no_grad():
                            embedding = self.vision_model(**processed_image)
                        page_embeddings[page_id] = embedding.cpu()
                        page_info[page_id] = {
                            "doc_id": doc_id,
                            "page_idx": page_idx,
                            "pdf_path": pdf_path,
                        }
                    except Exception as e:
                        logger.error(
                            f"Error processing page {page_idx} of PDF {pdf_path}: {str(e)}"
                        )
                        traceback.print_exc()
                        continue
            self._page_embeddings = page_embeddings
            self._page_info = page_info
            self._page_ids = list(page_embeddings.keys())
            self._visual_index_built = True

        # Nemo interactive index
        elif self.vision_retriever in ["nemo", "nvidia", "nemo_retriever"]:
            page_embeddings_list = []
            page_ids = []
            page_info = {}
            for doc_id, pdf_path in tqdm(
                pdf_paths.items(), desc="Processing PDFs for interactive visual index"
            ):
                try:
                    pages = convert_from_path(pdf_path)
                except Exception as e:
                    logger.error(
                        f"Error converting PDF {pdf_path} to images: {str(e)}"
                    )
                    traceback.print_exc()
                    continue

                if not pages:
                    continue

                try:
                    with torch.no_grad():
                        passage_embeddings = self.vision_model.forward_passages(
                            pages, batch_size=8
                        )
                    passage_embeddings = passage_embeddings.cpu()
                except Exception as e:
                    logger.error(
                        "Error encoding pages for interactive visual index with NemoRetriever: %s",
                        str(e),
                    )
                    traceback.print_exc()
                    continue

                for page_idx, page_img in enumerate(pages):
                    page_id = f"{doc_id}_{page_idx}"
                    page_ids.append(page_id)
                    page_embeddings_list.append(passage_embeddings[page_idx])
                    page_info[page_id] = {
                        "doc_id": doc_id,
                        "page_idx": page_idx,
                        "pdf_path": pdf_path,
                    }

            if not page_embeddings_list:
                logger.warning("No visual embeddings available for interactive index")
                self._visual_index_built = False
                return

            self._page_embeddings = torch.stack(page_embeddings_list, dim=0)
            self._page_ids = page_ids
            self._page_info = page_info
            self._visual_index_built = True
        else:
            raise ValueError(f"Unsupported visual retriever: {self.vision_retriever}")

    def retrieve_visual_contexts_for_question(self, query):
        try:
            self._build_interactive_visual_index()
            if not self._visual_index_built:
                return []
            # ColPali / ColQwen interactive retrieval
            if self.vision_retriever in ["colpali", "colqwen"]:
                if not self._page_embeddings:
                    return []
                page_ids = list(self._page_embeddings.keys())
                if not page_ids:
                    logger.warning(
                        "No visual embeddings available for interactive retrieval"
                    )
                    return []
                try:
                    processed_query = self.vision_processor.process_queries([query])
                    processed_query = {
                        k: v.to(self.vision_model.device)
                        for k, v in processed_query.items()
                    }
                    with torch.no_grad():
                        query_emb = self.vision_model(**processed_query)
                except Exception as e:
                    logger.error(
                        "Error generating query embedding for interactive visual retrieval: %s",
                        str(e),
                    )
                    traceback.print_exc()
                    return []
                ds = torch.cat(
                    [self._page_embeddings[pid] for pid in page_ids], dim=0
                )
                if len(ds) == 0:
                    logger.warning("No visual embeddings to score")
                    return []
                scores = self.vision_processor.score_multi_vector(query_emb, ds)
                scores = scores.flatten().numpy()
                top_indices = np.argsort(-scores)[: self.top_k]
                selected_page_ids = np.array(page_ids)[top_indices]
            # Nemo interactive retrieval
            elif self.vision_retriever in ["nemo", "nvidia", "nemo_retriever"]:
                if self._page_embeddings is None or not self._page_ids:
                    return []
                try:
                    with torch.no_grad():
                        query_emb = self.vision_model.forward_queries(
                            [query], batch_size=1
                        )
                    query_emb = query_emb.cpu()
                except Exception as e:
                    logger.error(
                        "Error generating query embedding for interactive visual retrieval: %s",
                        str(e),
                    )
                    traceback.print_exc()
                    return []

                try:
                    with torch.no_grad():
                        scores = self.vision_model.get_scores(
                            query_emb, self._page_embeddings
                        )
                    query_scores = scores[0].cpu().numpy()
                except Exception as e:
                    logger.error(
                        "Error scoring interactive visual retrieval with NemoRetriever: %s",
                        str(e),
                    )
                    traceback.print_exc()
                    return []

                if len(query_scores) == 0:
                    logger.warning("No visual embeddings to score")
                    return []

                top_indices = np.argsort(-query_scores)[: self.top_k]
                selected_page_ids = [
                    self._page_ids[idx]
                    for idx in top_indices
                    if 0 <= idx < len(self._page_ids)
                ]
            else:
                raise ValueError(f"Unsupported visual retriever: {self.vision_retriever}")

            pages = []
            for page_id in selected_page_ids:
                info = self._page_info.get(page_id)
                if not info:
                    continue
                pdf_path = info["pdf_path"]
                page_idx = info["page_idx"]
                try:
                    pdf_images = convert_from_path(pdf_path)
                    if page_idx >= len(pdf_images):
                        logger.warning(f"Page {page_idx} out of range for {pdf_path}")
                        continue
                    image = pdf_images[page_idx]
                    pages.append(
                        {
                            "image": image,
                            "document_id": page_id,
                            "page_number": page_idx,
                        }
                    )
                except Exception as e:
                    logger.error(f"Error loading PDF page for {page_id}: {str(e)}")
                    traceback.print_exc()
                    continue
            logger.info(
                "Retrieved %d visual contexts for interactive query", len(pages)
            )
            return pages
        except Exception as e:
            logger.error(
                "Error retrieving visual contexts for interactive query: %s", str(e)
            )
            traceback.print_exc()
            return []

    def encode_image(self, pil_image):
        """Encode a PIL image to base64 string."""
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str

    def generate_visual_response(self, query, visual_contexts):
        """Generate a response based on visual contexts."""
        try:
            # Extract just the images
            images = [ctx["image"] for ctx in visual_contexts]

            # Create prompt
            prompt_template = f"""
            You are tasked with answering a question based on the relevant pages of a PDF document. Provide your response in the following format:
            ## Evidence:

            ## Chain of Thought:

            ## Answer:

            ___
            Instructions:

            1. Evidence Curation: Extract relevant elements (such as paragraphs, tables, figures, charts) from the provided pages and populate them in the "Evidence" section. For each element, include the type, content, and a brief explanation of its relevance.

            2. Chain of Thought: In the "Chain of Thought" section, list out each logical step you take to derive the answer, referencing the evidence where applicable. You should perform computations if you need to to get to the answer. 

            3. Answer: {self.qa_prompt}
            ___
            Question: {query}
            """
            base64_images = [self.encode_image(img) for img in images]

            if self.llm_model == "gpt4":
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                },
                            }
                            for base64_image in base64_images
                        ]
                        + [
                            {"type": "text", "text": prompt_template},
                        ],
                    }
                ]

                response = self.client.chat.completions.create(
                    model="chatgpt-4o-latest",
                    messages=messages,
                    max_tokens=3000,
                    temperature=0.7,
                )

                return response.choices[0].message.content

            elif self.llm_model == "doubao":
                response = self.llm.responses.create(
                    model="doubao-seed-1-6-flash-250828",
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_image",
                                    "image_url": f"data:image/jpeg;base64,{base64_image}",
                                }
                                for base64_image in base64_images
                            ]
                            + [
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
                        "content": [
                            {"type": "image", "image": img} for img in images
                        ]
                        + [{"type": "text", "text": prompt_template}],
                    }
                ]

                text = self.qwen_processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, _ = self.process_vision_info(messages)
                inputs = self.qwen_processor(
                    text=[text], images=image_inputs, padding=True, return_tensors="pt"
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
            logger.error(f"Error generating visual response: {str(e)}")
            traceback.print_exc()
            return "Error generating response from visual contexts."


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

            # Prepare all documents
            all_chunks = []
            chunk_to_doc_mapping = []

            for doc_id, pages in tqdm(
                self.parent.document_cache.items(),
                desc="Processing documents for text index",
            ):
                all_text = "\n".join(pages)
                chunks = self.parent.split_text(all_text)

                for chunk in chunks:
                    all_chunks.append(chunk)
                    arxiv_id, page_num = self.parent.identify_document_and_page(chunk)
                    chunk_to_doc_mapping.append(
                        {
                            "chunk": chunk,
                            "chunk_pdf_name": arxiv_id if arxiv_id else doc_id,
                            "pdf_page_number": page_num if page_num is not None else 0,
                        }
                    )

            # Initialize retriever based on selected method
            if self.text_retriever == "bm25":
                # BM25 indexing
                bm25_model = BM25Okapi([chunk.split() for chunk in all_chunks])

                # Process each query
                results = []
                for _, row in tqdm(
                    self.df.iterrows(), desc="Processing queries for BM25"
                ):
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
                for _, row in tqdm(
                    self.df.iterrows(),
                    desc=f"Processing queries for {self.text_retriever.upper()}",
                ):
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
        # Build chunks per document and per page to avoid expensive identify_document_and_page
        for doc_id, pages in self.parent.document_cache.items():
            for page_idx, page_text in tqdm(
                enumerate(pages), total=len(pages), desc=f"Processing pages for {doc_id}"
            ):
                page_chunks = self.parent.split_text(page_text)
                for chunk in page_chunks:
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
                completion = self.llm.chat.completions.create(
                    model="doubao-1-5-lite-32k-250115",
                    messages=[
                        {"role": "user", "content": prompt_template},
                    ],
                )
                return completion.choices[0].message.content

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

# qa_prompts = {
#     "feta_tab": "You are a Wikipedia editor. Answer the question with a single, well-formed, factual sentence.",
#     "paper_tab": "You are a research scientist. Answer the question with a concise technical phrase.",
#     "scigraphqa": "You are a scientific researcher. Answer the question in 1-2 clear, evidence-based sentences.",
#     "slidevqa": "You are a presentation expert. Provide the exact answer to the question as it would appear on a slide. Be direct and precise.",
#     "spiqa": "You are a scientific paper author. Answer the question in 1-3 authoritative sentences."
# }

class VisDoMRAG:
    def __init__(self, config):
        """
        Initialize the VisDoMRAG pipeline.
        
        Args:
            config (dict): Configuration parameters
        """
        self.config = config
        self.data_dir = config["data_dir"]
        self.output_dir = config["output_dir"]
        self.llm_model = config["llm_model"]
        self.vision_retriever = config["vision_retriever"]
        self.text_retriever = config["text_retriever"]
        self.top_k = config.get("top_k", 5)
        self.api_keys = config.get("api_keys", {})
        self.chunk_size = config.get("chunk_size", 3000)
        self.chunk_overlap = config.get("chunk_overlap", 300)
        self.force_reindex = config.get("force_reindex", False)
        self.qa_prompt = config.get("qa_prompt", "Answer the question objectively based on the context provided.")
        self.pdf_files = config.get("pdf_files", [])
        self.dataset_csv = config.get("csv_path")
        if not self.dataset_csv:
            # Fallback to old behavior if csv_path not provided
            default_csv = f"{self.data_dir}/{os.path.basename(self.data_dir)}.csv"
            if os.path.exists(default_csv):
                self.dataset_csv = default_csv
        
        # Setup output directories
        os.makedirs(f"{self.output_dir}/{self.llm_model}_vision", exist_ok=True)
        os.makedirs(f"{self.output_dir}/{self.llm_model}_text", exist_ok=True)
        os.makedirs(f"{self.output_dir}/{self.llm_model}_visdmrag", exist_ok=True)
        
        # Create retrieval directories
        os.makedirs(f"{self.data_dir}/retrieval", exist_ok=True)
        
        # Initialize LLM
        self._initialize_llm()
        
        # Load dataset
        self.df = None
        if self.dataset_csv:
            logger.info(f"Loading dataset from {self.dataset_csv}")
            if not os.path.exists(self.dataset_csv):
                raise FileNotFoundError(f"CSV file not found: {self.dataset_csv}")
            self.df = pd.read_csv(self.dataset_csv)
        
        # Initialize document cache
        self.document_cache = {}
        
        # Initialize retrieval resources
        self._initialize_retrieval_resources()
        
    def _initialize_llm(self):
        """Initialize the LLM based on the selected model."""
        if self.llm_model == "doubao":
            if not self.api_keys.get("doubao"):
                raise ValueError("Doubao API key is required")
            self.llm = OpenAI(
                base_url="https://ark.cn-beijing.volces.com/api/v3",
                api_key=self.api_keys["doubao"],
            )
            logger.info("Initialized doubao model")
        
        elif self.llm_model == "gpt4":
            if not self.api_keys.get("openai"):
                raise ValueError("OpenAI API key is required")
            self.client = OpenAI(api_key=self.api_keys["openai"])
            logger.info("Initialized GPT-4 (via OpenAI client)")
        
        elif self.llm_model == "qwen":
            try:
                from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
                from qwen_vl_utils import process_vision_info
                
                self.qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
                    "Qwen/Qwen2-VL-7B-Instruct",
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                    device_map="auto",
                )
                min_pixels = 256*28*28
                max_pixels = 640*28*28
                self.qwen_processor = AutoProcessor.from_pretrained(
                    "Qwen/Qwen2-VL-7B-Instruct", 
                    min_pixels=min_pixels, 
                    max_pixels=max_pixels
                )
                self.process_vision_info = process_vision_info
                logger.info("Initialized Qwen2-VL model")
            except ImportError:
                raise ImportError("Required packages for Qwen not found. Install transformers and qwen_vl_utils.")
        else:
            raise ValueError(f"Unsupported LLM model: {self.llm_model}")
    
    def _initialize_retrieval_resources(self):
        """Initialize resources needed for retrieval."""
        # Create dedicated engines for visual and textual RAG
        self.visual_engine = VisualRAGEngine(self)
        self.textual_engine = TextualRAGEngine(self)

    def extract_text_from_pdf(self, pdf_path):
        """
        Extract text from a PDF file using OCR if needed.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            list: List of text from each page
        """
        try:
            # Use DotsOCR to extract structured text from PDF
            dots_output_dir = os.path.join(self.output_dir, "dots_ocr")
            os.makedirs(dots_output_dir, exist_ok=True)

            # Allow overriding DotsOCR parameters via config
            dots_kwargs = {
                "max_completion_tokens": self.config.get("dots_max_completion_tokens", 16384),
                "num_thread": self.config.get("dots_num_thread", 16),
                "dpi": self.config.get("dots_dpi", 200),
                "output_dir": dots_output_dir,
            }

            logger.info(f"Using DotsOCR to extract text from PDF: {pdf_path}")
            dots_ocr_parser = DotsOCRParser(**dots_kwargs)

            prompt_mode = self.config.get("dots_prompt_mode", "prompt_layout_all_en")
            results = dots_ocr_parser.parse_file(
                pdf_path,
                output_dir=dots_output_dir,
                prompt_mode=prompt_mode,
            )

            pages = []
            # Results are per page with 'page_no' and md_content_path/md_content_nohf_path
            for res in sorted(results, key=lambda r: r.get("page_no", 0)):
                md_path = res.get("md_content_path") or res.get("md_content_nohf_path")
                page_no = res.get("page_no", len(pages)) + 1
                page_text = ""
                if md_path and os.path.exists(md_path):
                    try:
                        with open(md_path, "r", encoding="utf-8") as f:
                            page_text = f.read()
                    except Exception as read_err:
                        logger.error(
                            f"Error reading DotsOCR markdown for {pdf_path} page {page_no}: {read_err}"
                        )
                pages.append(f"--- Page {page_no} ---\n{page_text}\n")

            return pages

        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path} with DotsOCR: {str(e)}")
            traceback.print_exc()
            return []
    
    def split_text(self, text):
        """
        Split text into chunks.
        
        Args:
            text (str): Text to split
            
        Returns:
            list: List of text chunks
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        return text_splitter.split_text(text)
    
    def _get_config_pdf_paths(self):
        paths = {}
        for entry in self.pdf_files:
            if os.path.isabs(entry):
                pdf_path = entry
            else:
                if "/" in entry:
                    pdf_path = os.path.join(self.data_dir, entry)
                else:
                    pdf_path = os.path.join(self.data_dir, entry)
            if not os.path.exists(pdf_path) and not entry.endswith(".pdf"):
                alt_entry = f"{entry}.pdf"
                if os.path.isabs(alt_entry):
                    alt_path = alt_entry
                else:
                    if "/" in alt_entry:
                        alt_path = os.path.join(self.data_dir, alt_entry)
                    else:
                        alt_path = os.path.join(self.data_dir, alt_entry)
                if os.path.exists(alt_path):
                    pdf_path = alt_path
            if not os.path.exists(pdf_path):
                logger.warning(f"PDF file not found: {pdf_path}")
                continue
            base = os.path.basename(pdf_path)
            doc_id = os.path.splitext(base)[0]
            if doc_id not in paths:
                paths[doc_id] = pdf_path
        return paths
    
    def cache_documents(self):
        """
        Cache document content for all PDFs in the dataset.
        
        Returns:
            dict: Dictionary mapping document IDs to text content
        """
        logger.info("Caching document content")
        
        try:
            cache = {}
            if self.pdf_files:
                pdf_paths = self._get_config_pdf_paths()
                for doc_id, pdf_path in tqdm(pdf_paths.items(), desc="Caching documents"):
                    cache[doc_id] = self.extract_text_from_pdf(pdf_path)
            elif self.df is not None:
                unique_docs = set()
                for _, row in self.df.iterrows():
                    try:
                        docs = eval(row['documents']) if 'documents' in row else []
                        unique_docs.update(docs)
                    except:
                    # Handle case where 'documents' field is not a valid list
                        traceback.print_exc()
                        pass
                    
                    if 'doc_path' in row:
                        doc_path = row['doc_path']
                        if isinstance(doc_path, str) and doc_path.strip():
                            unique_docs.add(os.path.basename(doc_path).split('.')[0])
                
                pdf_dir = os.path.join(self.data_dir, "docs")
                for doc_id in tqdm(unique_docs, desc="Caching documents"):
                # Try different possible filename formats
                    possible_paths = [
                        os.path.join(pdf_dir, doc_id),
                        os.path.join(pdf_dir, f"{doc_id}.pdf"),
                        os.path.join(pdf_dir, f"{doc_id.ljust(10, '0')}.pdf"),
                        os.path.join(pdf_dir, f"{doc_id.split('_')[0]}.pdf")
                    ]
                    
                    for pdf_path in possible_paths:
                        if os.path.exists(pdf_path):
                            cache[doc_id] = self.extract_text_from_pdf(pdf_path)
                            break
                    else:
                        logger.warning(f"No PDF file found for document {doc_id}")
            else:
                logger.warning("No dataset CSV or pdf_files provided for caching documents")
            
            self.document_cache = cache
            logger.info(f"Cached content for {len(cache)} documents")
            return cache
            
        except Exception as e:
            logger.error(f"Error caching documents: {str(e)}")
            traceback.print_exc()
            return {}
    
    def identify_document_and_page(self, chunk):
        """
        Identify which document and page a chunk belongs to.
        
        Args:
            chunk (str): Text chunk
            
        Returns:
            tuple: (arxiv_id, page_num)
        """
        max_ratio = 0
        best_match = (None, None)
        
        for arxiv_id, pages in self.document_cache.items():
            for page_num, page_text in enumerate(pages):
                ratio = SequenceMatcher(None, chunk, page_text).ratio()
                if ratio > max_ratio:
                    max_ratio = ratio
                    best_match = (arxiv_id, page_num)
        
        return best_match
    
    def build_visual_index(self):
        """Proxy to the visual RAG engine for building the visual index."""
        return self.visual_engine.build_visual_index()

    def build_text_index(self):
        """Proxy to the textual RAG engine for building the text index."""
        return self.textual_engine.build_text_index()

    def retrieve_visual_contexts(self, query_id):
        """
        Retrieve visual contexts using the specified visual retriever.
        
        Args:
            query_id (str): The query ID
            
        Returns:
            list: Top-k visual contexts (images)
        """
        return self.visual_engine.retrieve_visual_contexts(query_id)
            
    def retrieve_textual_contexts(self, query_id):
        """
        Retrieve textual contexts using the specified text retriever.
        
        Args:
            query_id (str): The query ID
            
        Returns:
            list: Top-k textual contexts

        """
        return self.textual_engine.retrieve_textual_contexts(query_id)

    def _build_interactive_visual_index(self):
        return self.visual_engine._build_interactive_visual_index()

    def retrieve_visual_contexts_for_question(self, query):
        return self.visual_engine.retrieve_visual_contexts_for_question(query)

    def _build_interactive_text_index(self):
        return self.textual_engine._build_interactive_text_index()

    def retrieve_textual_contexts_for_question(self, question):
        return self.textual_engine.retrieve_textual_contexts_for_question(question)

    def encode_image(self, pil_image):
        """Proxy to the visual RAG engine image encoder."""
        return self.visual_engine.encode_image(pil_image)

    def generate_visual_response(self, query, visual_contexts):
        """Proxy to the visual RAG engine for generating a visual-based response."""
        return self.visual_engine.generate_visual_response(query, visual_contexts)
    
    def generate_textual_response(self, query, textual_contexts):
        """Proxy to the textual RAG engine for generating a text-based response."""
        return self.textual_engine.generate_textual_response(query, textual_contexts)

    def extract_sections(self, text):
        """
        Extract sections from the generated text.
        
        Args:
            text (str): The generated text
            
        Returns:
            dict: Extracted sections
        """
        sections = {}
        headings = ["Evidence", "Chain of Thought", "Answer"]
        
        for i in range(len(headings)):
            heading = headings[i]
            next_heading = headings[i + 1] if i + 1 < len(headings) else None
            
            if next_heading:
                pattern = rf"## {heading}:(.*?)(?=## {next_heading}:)"
            else:
                pattern = rf"## {heading}:(.*)"
            
            match = re.search(pattern, text, re.DOTALL)
            if match:
                sections[heading] = match.group(1).strip()
            else:
                sections[heading] = ""
        
        return sections

    def combine_responses(self, query, visual_response, textual_response, answer=None):
        """
        Combine visual and textual responses to generate a final answer.
        
        Args:
            query (str): The user's question
            visual_response (dict): Response from visual contexts
            textual_response (dict): Response from textual contexts
            answer (str): Ground truth answer if available
            
        Returns:
            dict: Combined response
        """
        try:
            prompt = f"""
            Analyze the following two responses to the question: "{query}"

            Response 1:
            Evidence: {visual_response.get('Evidence', "Evidence not available")}
            Chain of Thought: {visual_response.get('Chain of Thought', "CoT not available")}
            Final Answer: {visual_response['Answer']}

            Response 2:
            Evidence: {textual_response.get('Evidence', "Evidence not available")}
            Chain of Thought: {textual_response.get('Chain of Thought', "CoT not available")}
            Final Answer: {textual_response['Answer']}

            Response 1 is based on a visual q/a pipeline, and Response 2 is based on a textual q/a pipeline. 
            - In general, given both response 1 and response 2 have logical chains of thoughts, and decision boils down to evidence, you should place higher degree of trust on evidence reported in Response 1.
            - If one of the responses has declined giving a clear answer, please weigh the other answer more unless there is reasonable thought to not answer, and both thoughts are inconsistent.
            - Language of the answer should be short and direct, usually answerable in a single sentence, or phrase. You should directly give the specific response to an answer.

            Consider both chains of thought and final answers. Provide your analysis in the following format:

            ## Analysis:
            [Your detailed analysis here, evaluating the consistency of both the chains of thoughts, with respect to each other, the question and their respective answers, as well as validity of the evidence.]

            ## Conclusion:
            [Your conclusion on which answer is more likely to be correct, or if a synthesis of both is needed]

            ## Final Answer:
            [Answer the question "{query}", based on your analysis of the two candidates so far. Please ensure that answers are short and concise, similar in language to the provided answers.]
            """
            
            if self.llm_model == "gpt4":
                response = self.client.chat.completions.create(
                    model="chatgpt-4o-latest",
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1500,
                    temperature=0.3
                )
                return self.parse_combined_output(response.choices[0].message.content)
                
            elif self.llm_model == "doubao":
                combined_response = self.llm.chat.completions.create(
                    model="doubao-1-5-lite-32k-250115",
                    messages=[
                         {"role": "user", "content": prompt},
                    ],
                )
                return self.parse_combined_output(combined_response.choices[0].message.content)
                
            elif self.llm_model == "qwen":
                messages = [
                    {"role": "user", "content": [{"type": "text", "text": prompt}]}
                ]
                
                text = self.qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.qwen_processor(text=[text], padding=True, return_tensors="pt").to("cuda:0")

                generated_ids = self.qwen_model.generate(**inputs, max_new_tokens=1000)
                generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
                output_text = self.qwen_processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
                
                return self.parse_combined_output(output_text[0])
        
        except Exception as e:
            logger.error(f"Error combining responses: {str(e)}")
            return {
                "Analysis": "Error occurred during analysis.",
                "Conclusion": "Error occurred during conclusion.",
                "Final Answer": "Error occurred during combination of responses."
            }

    def parse_combined_output(self, output):
        """
        Parse the output of the combination step.
        
        Args:
            output (str): The combined output text
            
        Returns:
            dict: Parsed sections
        """
        sections = {'Analysis': '', 'Conclusion': '', 'Final Answer': ''}
        current_section = None

        for line in output.split('\n'):
            if line.startswith('## '):
                current_section = line[3:].strip(':')
            elif current_section and current_section in sections:
                sections[current_section] += line + '\n'

        # Clean up the sections
        for key in sections:
            sections[key] = sections[key].strip()

        return sections

    def answer_question(self, question):
        try:
            start_time = time.time()

            logger.info("Step 1/5: Retrieving visual contexts...")
            step_start = time.time()
            visual_contexts = self.retrieve_visual_contexts_for_question(question)
            logger.info("Step 1/5 completed in %.2f seconds", time.time() - step_start)
            visual_response_dict = None
            if visual_contexts:
                logger.info("Step 2/5: Generating visual response...")
                step_start = time.time()
                visual_response = self.generate_visual_response(question, visual_contexts)
                logger.info("Step 2/5 completed in %.2f seconds", time.time() - step_start)
                visual_response_dict = self.extract_sections(visual_response)
                visual_response_dict.update(
                    {
                        "question": question,
                        "document": [ctx["document_id"] for ctx in visual_contexts],
                        "pages": [ctx["page_number"] for ctx in visual_contexts],
                    }
                )
            logger.info("Step 3/5: Retrieving textual contexts...")
            step_start = time.time()
            textual_contexts = self.retrieve_textual_contexts_for_question(question)
            logger.info("Step 3/5 completed in %.2f seconds", time.time() - step_start)
            textual_response_dict = None
            if textual_contexts:
                logger.info("Step 4/5: Generating textual response...")
                step_start = time.time()
                textual_response = self.generate_textual_response(question, textual_contexts)
                logger.info("Step 4/5 completed in %.2f seconds", time.time() - step_start)
                textual_response_dict = self.extract_sections(textual_response)
                textual_response_dict.update(
                    {
                        "question": question,
                        "document": [ctx["chunk_pdf_name"] for ctx in textual_contexts],
                        "pages": [ctx["pdf_page_number"] for ctx in textual_contexts],
                        "chunks": "\n".join([ctx["chunk"] for ctx in textual_contexts]),
                    }
                )
            if visual_response_dict and textual_response_dict:
                logger.info("Step 5/5: Combining responses...")
                step_start = time.time()
                combined_sections = self.combine_responses(
                    question,
                    visual_response_dict,
                    textual_response_dict,
                    None
                )
                logger.info("Step 5/5 completed in %.2f seconds", time.time() - step_start)
                combined_response = {
                    "question": question,
                    "answer": combined_sections.get("Final Answer", ""),
                    "analysis": combined_sections.get("Analysis", ""),
                    "conclusion": combined_sections.get("Conclusion", ""),
                    "response1": visual_response_dict,
                    "response2": textual_response_dict,
                }
                logger.info("answer_question completed in %.2f seconds", time.time() - start_time)
                return combined_response
            if visual_response_dict:
                logger.info("answer_question completed in %.2f seconds (visual only)", time.time() - start_time)
                return {
                    "question": question,
                    "answer": visual_response_dict.get("Answer", ""),
                    "analysis": visual_response_dict.get("Chain of Thought", ""),
                    "conclusion": "",
                    "response1": visual_response_dict,
                    "response2": textual_response_dict,
                }
            if textual_response_dict:
                logger.info("answer_question completed in %.2f seconds (textual only)", time.time() - start_time)
                return {
                    "question": question,
                    "answer": textual_response_dict.get("Answer", ""),
                    "analysis": textual_response_dict.get("Chain of Thought", ""),
                    "conclusion": "",
                    "response1": visual_response_dict,
                    "response2": textual_response_dict,
                }
            logger.warning("Missing responses for interactive question")
            return None
        except Exception as e:
            logger.error(f"Error answering interactive question: {str(e)}")
            traceback.print_exc()
            return None

    def process_query(self, query_id):
        """
        Process a single query through the complete VisDoMRAG pipeline.
        
        Args:
            query_id (str): The query ID
            
        Returns:
            bool: Success status
        """
        try:
            if self.df is None:
                raise ValueError("Dataset CSV not loaded; process_query is only available in benchmark mode.")
            # Get query information
            query_row = self.df[self.df['q_id'] == query_id].iloc[0]
            question = query_row['question']
            
            try:
                # Try to parse the answer field as a list/dict if it's in that format
                answer = eval(query_row['answer'])
            except:
                # If parsing fails, use as-is
                answer = query_row['answer']
            
            # Define file paths for outputs
            visual_file = f"{self.output_dir}/{self.llm_model}_vision/response_{str(query_id).replace('/','$')}.json"
            textual_file = f"{self.output_dir}/{self.llm_model}_text/response_{str(query_id).replace('/','$')}.json"
            combined_file = f"{self.output_dir}/{self.llm_model}_visdmrag/response_{str(query_id).replace('/','$')}.json"
            
            # Skip if the combined file already exists
            if os.path.exists(combined_file):
                logger.info(f"Combined file already exists for query {query_id}")
                return True
            
            # Process visual contexts if needed
            visual_response_dict = None
            if not os.path.exists(visual_file):
                logger.info(f"Generating visual response for query {query_id}")
                visual_contexts = self.retrieve_visual_contexts(query_id)
                
                if visual_contexts:
                    visual_response = self.generate_visual_response(question, visual_contexts)
                    visual_response_dict = self.extract_sections(visual_response)
                    
                    # Add metadata
                    visual_response_dict.update({
                        "question": question,
                        "document": [ctx['document_id'] for ctx in visual_contexts],
                        "gt_answer": answer,
                        "pages": [ctx['page_number'] for ctx in visual_contexts]
                    })
                    
                    # Save visual response
                    with open(visual_file, 'w') as file:
                        json.dump(visual_response_dict, file, indent=4)
                    
                    # Memory cleanup
                    del visual_contexts
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            else:
                # Load existing visual response
                with open(visual_file, 'r') as file:
                    visual_response_dict = json.load(file)
            
            # Process textual contexts if needed
            textual_response_dict = None
            if not os.path.exists(textual_file):
                logger.info(f"Generating textual response for query {query_id}")
                textual_contexts = self.retrieve_textual_contexts(query_id)
                
                if textual_contexts:
                    textual_response = self.generate_textual_response(question, textual_contexts)
                    textual_response_dict = self.extract_sections(textual_response)
                    
                    # Add metadata
                    textual_response_dict.update({
                        "question": question,
                        "document": [ctx['chunk_pdf_name'] for ctx in textual_contexts],
                        "gt_answer": answer,
                        "pages": [ctx['pdf_page_number'] for ctx in textual_contexts],
                        "chunks": "\n".join([ctx['chunk'] for ctx in textual_contexts])
                    })
                    
                    # Save textual response
                    with open(textual_file, 'w') as file:
                        json.dump(textual_response_dict, file, indent=4)
                    
                    # Memory cleanup
                    del textual_contexts
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            else:
                # Load existing textual response
                with open(textual_file, 'r') as file:
                    textual_response_dict = json.load(file)
            
            # Skip if either response is missing
            if not visual_response_dict or not textual_response_dict:
                logger.warning(f"Missing responses for query {query_id}")
                return False
            
            # Combine responses
            logger.info(f"Combining responses for query {query_id}")
            combined_sections = self.combine_responses(
                question, 
                visual_response_dict, 
                textual_response_dict,
                answer
            )
            
            # Create combined response
            combined_response = {
                "question": question,
                "answer": combined_sections.get("Final Answer", ""),
                "gt_answer": answer,
                "analysis": combined_sections.get("Analysis", ""),
                "conclusion": combined_sections.get("Conclusion", ""),
                "response1": visual_response_dict,
                "response2": textual_response_dict
            }
            
            # Save combined response
            with open(combined_file, 'w') as file:
                json.dump(combined_response, file, indent=4)
            
            # Memory cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing query {query_id}: {str(e)}")
            return False

    def run(self):
        """Run the VisDoMRAG pipeline on all queries in the dataset."""
        logger.info("Starting VisDoMRAG pipeline")
        
        if self.df is None:
            logger.warning("No dataset CSV loaded; run() is available only in benchmark mode.")
            return
        
        # Process each query
        for query_id in tqdm(self.df['q_id'].unique()):
            try:
                logger.info(f"Processing query {query_id}")
                success = self.process_query(query_id)
                
                if success:
                    logger.info(f"Successfully processed query {query_id}")
                else:
                    logger.warning(f"Failed to process query {query_id}")
                
                # Pause to avoid hitting API rate limits
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error processing query {query_id}: {str(e)}")
        
        logger.info("VisDoMRAG pipeline completed")