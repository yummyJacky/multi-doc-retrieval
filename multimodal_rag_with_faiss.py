"""
Nvidia NemoRetriever RAG Pipeline with FAISS Indexing
重构版本：添加向量索引支持，提升大规模文档检索性能

融合mmdocir思路：支持混合检索（图像/布局 + 文本）
"""

import os
import gc
import torch
import numpy as np
import faiss
from typing import List, Dict, Tuple, Optional
from PIL import Image
from pathlib import Path
import pickle
from tqdm.auto import tqdm
from openai import OpenAI

class GPUMemoryMonitor:
    """GPU显存监控工具"""
    
    def __init__(self, device: int = 0):
        self.device = device
    
    def print_memory(self, stage: str = ""):
        """打印当前GPU显存使用情况"""
        if torch.cuda.is_available():
            torch.cuda.set_device(self.device)
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"[{stage}] GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    
    @staticmethod
    def clear_memory():
        """清理GPU显存"""
        gc.collect()
        torch.cuda.empty_cache()


class FAISSIndexManager:
    """FAISS索引管理器"""
    
    def __init__(
        self,
        embedding_dim: int,
        index_type: str = "flat",
        use_gpu: bool = False,
        gpu_id: int = 0
    ):
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        self.index = None
        self.is_trained = False

    def build_index(self, embeddings: np.ndarray):
        print(f"Building FAISS index (type: {self.index_type}) with {len(embeddings)} vectors.")
        if self.index_type == "flat":
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")

        if self.use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, self.gpu_id, self.index)

        self.index.add(embeddings)
        self.is_trained = True
        print("FAISS index built successfully.")

    def search(self, query_embeddings: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_trained:
            raise RuntimeError("Index is not built or trained.")
        return self.index.search(query_embeddings, top_k)

    def save(self, path: str):
        index_to_save = faiss.index_gpu_to_cpu(self.index) if self.use_gpu else self.index
        faiss.write_index(index_to_save, path)

    def load(self, path: str):
        self.index = faiss.read_index(path)
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, self.gpu_id, self.index)
        self.is_trained = True

class NvidiaRAGPipeline:
    """Nvidia NemoRetriever RAG Pipeline with Hybrid FAISS Indices"""
    
    def __init__(
        self,
        retriever_model,
        reranker=None,
        vqa_model=None,
        device: int = 0,
        faiss_gpu: bool = False
    ):
        self.retriever_model = retriever_model
        self.reranker = reranker
        self.vqa_model = vqa_model
        self.device = device
        self.memory_monitor = GPUMemoryMonitor(device)
        
        embedding_dim = retriever_model.config.hidden_size
        
        # 图像/布局索引
        self.image_faiss = FAISSIndexManager(embedding_dim, index_type="flat", use_gpu=faiss_gpu, gpu_id=device)
        self.image_passage_ids = []

        # 文本/OCR索引
        self.text_faiss = FAISSIndexManager(embedding_dim, index_type="flat", use_gpu=faiss_gpu, gpu_id=device)
        self.text_passage_ids = []
        self.text_chunks = []

    def _encode_batch(self, items, encode_fn, batch_size):
        all_embeddings = []
        for i in tqdm(range(0, len(items), batch_size), desc=f"Encoding {encode_fn.__name__}"):
            batch = items[i:i+batch_size]
            with torch.no_grad():
                embeddings = encode_fn(batch).cpu()
            all_embeddings.append(embeddings)
        return torch.cat(all_embeddings, dim=0)

    def build_indices(
        self, 
        images: List[Image.Image], 
        ocr_texts: List[str], 
        page_ids: List[str],
        batch_size: int = 8,
        save_dir: Optional[str] = None
    ):
        print("Building hybrid indices...")
        self.memory_monitor.print_memory("Start index building")

        # 1. Build Image/Layout Index
        print("Encoding images for layout index...")
        image_embeddings = self._encode_batch(images, self.retriever_model.forward_passages, batch_size)
        image_embeddings_np = image_embeddings.numpy().astype(np.float32)
        faiss.normalize_L2(image_embeddings_np)
        self.image_faiss.build_index(image_embeddings_np)
        self.image_passage_ids = page_ids
        self.memory_monitor.print_memory("After image encoding")

        # 2. Build Text/OCR Index
        print("Encoding OCR texts for text index...")
        # Simple chunking for demonstration
        self.text_chunks = [chunk for text in ocr_texts for chunk in (text.split('\n\n') if text else [''])]
        self.text_passage_ids = [pid for pid, text in zip(page_ids, ocr_texts) for _ in (text.split('\n\n') if text else [''])]
        
        text_embeddings = self._encode_batch(self.text_chunks, self.retriever_model.forward_queries, batch_size)
        text_embeddings_np = text_embeddings.numpy().astype(np.float32)
        faiss.normalize_L2(text_embeddings_np)
        self.text_faiss.build_index(text_embeddings_np)
        self.memory_monitor.print_memory("After text encoding")

        if save_dir:
            self.save_indices(save_dir)

    def retrieve(self, query: str, top_k: int = 10, image_weight: float = 0.5, text_weight: float = 0.5) -> List[Dict]:
        print(f"Performing hybrid retrieval for query: '{query}'")
        with torch.no_grad():
            query_embedding = self.retriever_model.forward_queries([query]).cpu().numpy().astype(np.float32)
        faiss.normalize_L2(query_embedding)

        # Search image index
        image_scores, image_indices = self.image_faiss.search(query_embedding, top_k)
        
        # Search text index
        text_scores, text_indices = self.text_faiss.search(query_embedding, top_k)

        # Weighted fusion of results
        fused_scores = {}

        for i in range(top_k):
            # Image results
            img_idx = image_indices[0, i]
            img_pid = self.image_passage_ids[img_idx]
            img_score = image_scores[0, i]
            if img_pid not in fused_scores:
                fused_scores[img_pid] = 0
            fused_scores[img_pid] += img_score * image_weight

            # Text results
            txt_idx = text_indices[0, i]
            txt_pid = self.text_passage_ids[txt_idx]
            txt_score = text_scores[0, i]
            if txt_pid not in fused_scores:
                fused_scores[txt_pid] = 0
            fused_scores[txt_pid] += txt_score * text_weight

        # Sort by fused score
        sorted_results = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
        
        final_results = []
        for page_id, score in sorted_results[:top_k]:
            final_results.append({
                "page_id": page_id,
                "score": score
            })
        
        return final_results

    def save_indices(self, save_dir: str):
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save image index
        self.image_faiss.save(str(path / "image.index"))
        with open(path / "image_passage_ids.pkl", "wb") as f:
            pickle.dump(self.image_passage_ids, f)
            
        # Save text index
        self.text_faiss.save(str(path / "text.index"))
        with open(path / "text_passage_ids.pkl", "wb") as f:
            pickle.dump(self.text_passage_ids, f)
        with open(path / "text_chunks.pkl", "wb") as f:
            pickle.dump(self.text_chunks, f)
            
        print(f"Indices saved to {save_dir}")

    def load_indices(self, load_dir: str):
        path = Path(load_dir)
        
        # Load image index
        self.image_faiss.load(str(path / "image.index"))
        with open(path / "image_passage_ids.pkl", "rb") as f:
            self.image_passage_ids = pickle.load(f)

        # Load text index
        self.text_faiss.load(str(path / "text.index"))
        with open(path / "text_passage_ids.pkl", "rb") as f:
            self.text_passage_ids = pickle.load(f)
        with open(path / "text_chunks.pkl", "rb") as f:
            self.text_chunks = pickle.load(f)
            
        print(f"Indices loaded from {load_dir}")

class ImageReranker:
    """图片重排序器（使用 MonoVLM）"""
    
    def __init__(self, model_name: str = "monovlm", device: str = "cuda:0", use_fast: bool = True):
        try:
            from rerankers import Reranker
        except ImportError:
            raise ImportError("请安装 rerankers 库: pip install rerankers")
        
        self.device = device
        print(f"Loading {model_name} reranker on {device}...")
        self.ranker = Reranker(model_name, use_fast=use_fast, device=device)
        print("? Reranker loaded successfully")
    
    def rerank(
        self,
        query: str,
        images: List[Image.Image],
        top_k: int = 5
    ) -> List[Dict]:
        results = self.ranker.rank(
            query=query,
            images=images,
        )
        return results[:top_k]

    def rerank_batch(
        self,
        queries: List[str],
        all_images_list: List[List[Image.Image]],
        top_k: int = 5
    ) -> List[List[Dict]]:
        
        all_results = []
        for query, images in tqdm(zip(queries, all_images_list), total=len(queries), desc="Reranking queries"):
            reranked = self.rerank(query, images, top_k)
            all_results.append(reranked)
        return all_results

class VQAModel:
    """VQA模型，用于最终问答"""

    def __init__(self, model_name: str, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "your-api-key")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        print(f"? VQA Model initialized with API: {self.model_name}")

    def _image_to_base64(self, image: Image.Image) -> str:
        import base64
        from io import BytesIO
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def answer_with_multiple_images(
        self, 
        query: str, 
        images: List[Image.Image], 
        max_tokens: int = 1024
    ) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                ] + [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{self._image_to_base64(img)}"}
                    } for img in images
                ]
            }
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.0,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in VQA API call: {e}")
            return "Sorry, I couldn't process the answer."

    def answer_batch_with_multiple_images(
        self, 
        queries: List[str], 
        all_images_list: List[List[Image.Image]],
        max_tokens: int = 1024
    ):
        answers = []
        for i, (query, images) in enumerate(tqdm(zip(queries, all_images_list), total=len(queries), desc="VQA Processing (Multi-Image)")):
            answer = self.answer_with_multiple_images(query, images, max_tokens)
            answers.append(answer)
            
            # Display intermediate results
            print("\n" + "="*80)
            print(f"问题 {i+1}: {query}")
            print("="*80)
            print(f"使用页面: Top-{len(images)} 综合分析")
            # Assuming images have a 'filename' or similar attribute if you want to show which pages were used
            # This part needs adjustment based on how you track image origins.
            print("\n答案:")
            print(answer)
        
        return answers
