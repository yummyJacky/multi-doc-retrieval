import os
import csv
import base64
import logging
import traceback
from io import BytesIO

import numpy as np
import pandas as pd
import torch
from pdf2image import convert_from_path
from tqdm import tqdm
from transformers import AutoModel


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


__all__ = ["VisualRAGEngine"]
