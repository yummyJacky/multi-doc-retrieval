import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
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
import traceback
from openai import OpenAI
from dots_ocr.parser import DotsOCRParser
from transformers import AutoModel, AutoTokenizer
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
import shutil
from visual_rag import VisualRAGEngine 
from textual_rag import TextualRAGEngine 
from ocr_extractor import OCRExtractor
from qwenvl_caption import QwenVLCaptioner

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
        self.ocr_engine = config.get("ocr_engine", "dots")  # 'dots' or 'deepseek'
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
        
        # Helper modules for OCR extraction and Qwen-VL image captioning
        self.ocr_extractor = OCRExtractor(self.config, self.output_dir, logger)
        self.qwen_captioner = QwenVLCaptioner(self.config.get("qwen_vl_checkpoint"), logger)
        
        # Retrieval engines (lazy init when first used)
        self.visual_engine = None
        self.textual_engine = None
        
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

    def _initialize_visual_retrieval_resources(self):
        """Lazily initialize resources needed for retrieval (visual & textual)."""
        if self.visual_engine is None:
            self.visual_engine = VisualRAGEngine(self)
            
    def _initialize_textual_retrieval_resources(self):
        if self.textual_engine is None:
            self.textual_engine = TextualRAGEngine(self)

    def extract_text_from_pdf(self, pdf_path):
        """
        Extract text from a PDF file using OCR if needed.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            list: List of text from each page
        """
        # Allow switching between DotsOCR and DeepSeek-OCR via config
        ocr_engine = self.ocr_engine.lower() if isinstance(self.ocr_engine, str) else "dots"

        # Use external OCRExtractor and (optionally) Qwen-VL captioner to
        # perform OCR and image caption augmentation on the markdown.
        captioner = None
        if self.config.get("qwen_vl_checkpoint"):
            captioner = self.qwen_captioner

        return self.ocr_extractor.extract_text_from_pdf(
            pdf_path,
            ocr_engine=ocr_engine,
            captioner=captioner,
        )
    
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
    
    def build_visual_index(self):
        """Proxy to the visual RAG engine for building the visual index."""
        self._initialize_visual_retrieval_resources()
        return self.visual_engine.build_visual_index()

    def build_text_index(self):
        """Proxy to the textual RAG engine for building the text index."""
        self._initialize_textual_retrieval_resources()
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
                # combined_response = self.llm.chat.completions.create(
                #     model="doubao-1-5-lite-32k-250115",
                #     messages=[
                #          {"role": "user", "content": prompt},
                #     ],
                # )
                # return self.parse_combined_output(combined_response.choices[0].message.content)
                combined_response = self.llm.responses.create(
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
                return self.parse_combined_output(combined_response.output[1].content[0].text)
                
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
            visual_response_dict = None
            textual_response_dict = None

            def _run_visual_branch():
                logger.info("Step 1/5: Retrieving visual contexts...")
                step_start = time.time()
                visual_contexts_local = self.retrieve_visual_contexts_for_question(question)
                logger.info(
                    "Step 1/5 completed in %.2f seconds", time.time() - step_start
                )
                if not visual_contexts_local:
                    return None
                logger.info("Step 2/5: Generating visual response...")
                step_start_inner = time.time()
                visual_response = self.generate_visual_response(
                    question, visual_contexts_local
                )
                logger.info(
                    "Step 2/5 completed in %.2f seconds",
                    time.time() - step_start_inner,
                )
                vr_dict = self.extract_sections(visual_response)
                vr_dict.update(
                    {
                        "question": question,
                        "document": [ctx["document_id"] for ctx in visual_contexts_local],
                        "pages": [ctx["page_number"] for ctx in visual_contexts_local],
                    }
                )
                return vr_dict

            def _run_textual_branch():
                logger.info("Step 3/5: Retrieving textual contexts...")
                step_start = time.time()
                textual_contexts_local = self.retrieve_textual_contexts_for_question(
                    question
                )
                logger.info(
                    "Step 3/5 completed in %.2f seconds", time.time() - step_start
                )
                if not textual_contexts_local:
                    return None
                for i, ctx in enumerate(textual_contexts_local):
                    chunk = ctx.get("chunk", "")
                    doc_id = ctx.get("chunk_pdf_name")
                    page_num = ctx.get("pdf_page_number")
                    preview = chunk[:60].replace("\n", " ") + ("..." if len(chunk) > 60 else "")
                    print(f"  - #{i}: doc={doc_id}, page={page_num}, text_preview={preview}")
                logger.info("Step 4/5: Generating textual response...")
                step_start_inner = time.time()
                textual_response = self.generate_textual_response(
                    question, textual_contexts_local
                )
                logger.info(
                    "Step 4/5 completed in %.2f seconds",
                    time.time() - step_start_inner,
                )
                tr_dict = self.extract_sections(textual_response)
                tr_dict.update(
                    {
                        "question": question,
                        "document": [
                            ctx["chunk_pdf_name"] for ctx in textual_contexts_local
                        ],
                        "pages": [
                            ctx["pdf_page_number"] for ctx in textual_contexts_local
                        ],
                        "chunks": "\n".join(
                            [ctx["chunk"] for ctx in textual_contexts_local]
                        ),
                    }
                )
                return tr_dict

            # Run visual and textual branches in parallel
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_to_kind = {
                    executor.submit(_run_visual_branch): "visual",
                    executor.submit(_run_textual_branch): "textual",
                }
                for future in as_completed(future_to_kind):
                    kind = future_to_kind[future]
                    try:
                        result = future.result()
                    except Exception as e:
                        logger.error(
                            "Error in %s branch while answering question: %s",
                            kind,
                            str(e),
                        )
                        traceback.print_exc()
                        continue
                    if kind == "visual":
                        visual_response_dict = result
                    else:
                        textual_response_dict = result

            if visual_response_dict and textual_response_dict:
                logger.info("Step 5/5: Combining responses...")
                step_start = time.time()
                combined_sections = self.combine_responses(
                    question,
                    visual_response_dict,
                    textual_response_dict,
                    None,
                )
                logger.info(
                    "Step 5/5 completed in %.2f seconds", time.time() - step_start
                )
                combined_response = {
                    "question": question,
                    "answer": combined_sections.get("Final Answer", ""),
                    "analysis": combined_sections.get("Analysis", ""),
                    "conclusion": combined_sections.get("Conclusion", ""),
                    "response1": visual_response_dict,
                    "response2": textual_response_dict,
                }
                logger.info(
                    "answer_question completed in %.2f seconds", time.time() - start_time
                )
                return combined_response
            if visual_response_dict:
                logger.info(
                    "answer_question completed in %.2f seconds (visual only)",
                    time.time() - start_time,
                )
                return {
                    "question": question,
                    "answer": visual_response_dict.get("Answer", ""),
                    "analysis": visual_response_dict.get("Chain of Thought", ""),
                    "conclusion": "",
                    "response1": visual_response_dict,
                    "response2": textual_response_dict,
                }
            if textual_response_dict:
                logger.info(
                    "answer_question completed in %.2f seconds (textual only)",
                    time.time() - start_time,
                )
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