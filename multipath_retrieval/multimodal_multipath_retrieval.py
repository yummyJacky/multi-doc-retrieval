"""
多模态多路召回系统
结合MetaCLIP2和Nvidia FAISS的多路召回，并融合结果

主要功能：
1. MetaCLIP2图像-文本检索
2. Nvidia FAISS文档检索  
3. 多路召回结果融合
4. 统一的检索接口
"""
import os
import logging
import time
from dataclasses import dataclass
from typing import List, Optional

import faiss
import numpy as np
import torch
from PIL import Image
from pdf2image import convert_from_path
from transformers import AutoModel, AutoProcessor
from dotenv import load_dotenv
load_dotenv()
# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    """检索结果数据类"""
    page_num: int
    page_index: int
    score: float
    source: str  # 'metaclip' or 'nvidia_faiss'
    content_type: str  # 'image' or 'document'
    
@dataclass
class FusedResult:
    """融合后的检索结果"""
    page_num: int
    page_index: int
    final_score: float
    metaclip_score: Optional[float] = None
    nvidia_score: Optional[float] = None
    fusion_method: str = "weighted_sum"

class MetaCLIPRetriever:
    """MetaCLIP2图像检索器"""
    
    def __init__(self, device: str = "cuda:0", model_name: str = "facebook/metaclip-2-worldwide-giant"):
        self.device = device
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.index = None
        self.images = None
        self.embedding_dim = 1280
        
    def load_model(self, hf_token: Optional[str] = None):
        """加载MetaCLIP2模型"""
        logger.info(f"Loading MetaCLIP2 model on {self.device}")
        
        self.model = AutoModel.from_pretrained(
            self.model_name,
            dtype=torch.float32,
            attn_implementation="sdpa",
            token=hf_token
        ).to(self.device)
        
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            token=hf_token
        )
        
        logger.info("MetaCLIP2 model loaded successfully")
        
    def build_image_index(self, images: List[Image.Image], save_path: Optional[str] = None):
        """构建图像索引"""
        logger.info(f"Building image index for {len(images)} images")
        
        self.images = images
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        # 批量编码图像
        for i, image in enumerate(images):
            embedding = self._encode_image(image)
            self._add_to_index(embedding)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(images)} images")
        
        if save_path:
            faiss.write_index(self.index, save_path)
            logger.info(f"Index saved to {save_path}")
            
        logger.info("Image index built successfully")
        
    def _encode_image(self, image: Image.Image) -> np.ndarray:
        """编码单张图像"""
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            image_features = self.model.get_image_features(**inputs)
            
            # 转换为numpy并归一化
            embedding = image_features.detach().cpu().numpy().astype(np.float32)
            faiss.normalize_L2(embedding)
            return embedding
            
    def _add_to_index(self, embedding: np.ndarray):
        """添加向量到索引"""
        self.index.add(embedding)
        
    def search(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """文本查询图像"""
        if self.index is None:
            raise ValueError("Index not built. Call build_image_index first.")
            
        # 编码查询文本
        text_token = self.processor.tokenizer([query], return_tensors="pt").to(self.device)
        text_features = self.model.get_text_features(**text_token)
        
        # 转换为numpy并归一化
        text_embedding = text_features.detach().cpu().numpy().astype(np.float32)
        faiss.normalize_L2(text_embedding)
        
        # 搜索
        distances, indices = self.index.search(text_embedding, top_k)
        
        # 转换为RetrievalResult
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx >= 0:  # 有效索引
                results.append(RetrievalResult(
                    page_num=idx + 1,  # 页码从1开始
                    page_index=idx,
                    score=float(1.0 / (1.0 + distance)),  # 距离转换为相似度分数
                    source="metaclip",
                    content_type="image"
                ))
        
        return results

class NvidiaFAISSRetriever:
    """Nvidia FAISS文档检索器"""
    
    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self.rag_pipeline = None
        self.images = None
        
    def load_model(self, model_name: str = "nvidia/llama-nemoretriever-colembed-3b-v1"):
        """加载Nvidia检索模型"""
        logger.info(f"Loading Nvidia model on {self.device}")
        
        from transformers import AutoModel
        
        retriever_model = AutoModel.from_pretrained(
            model_name,
            device_map=f'cuda:{self.device.split(":")[-1]}',
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            revision='50c36f4d5271c6851aa08bd26d69f6e7ca8b870c',
        ).eval()
        
        # 导入自定义的RAG Pipeline
        try:
            from nvidia_rag_with_faiss import NvidiaRAGPipeline
            
            faiss_config = {
                "embedding_dim": 3072,
                "index_type": "ivfflat",
                "nlist": 100,  # 默认聚类数
                "use_gpu": True,
            }
            
            self.rag_pipeline = NvidiaRAGPipeline(
                retriever_model=retriever_model,
                use_faiss=True,
                faiss_config=faiss_config,
                device=self.device.split(":")[-1]
            )
            
            logger.info("Nvidia FAISS model loaded successfully")
            
        except ImportError:
            logger.error("nvidia_rag_with_faiss module not found. Please ensure it's available.")
            raise
            
    def build_document_index(self, images: List[Image.Image], save_dir: Optional[str] = None):
        """构建文档索引"""
        logger.info(f"Building document index for {len(images)} pages")
        
        self.images = images
        
        # 编码文档
        passage_embeddings, page_ids = self.rag_pipeline.encode_passages(
            images=images,
            batch_size=8,
            page_id_prefix="multimodal_doc"
        )
        
        # 构建或加载索引
        if save_dir:
            self.rag_pipeline.build_index(save_dir=save_dir)
        
        logger.info("Document index built successfully")
        
    def search(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """文档检索"""
        if self.rag_pipeline is None:
            raise ValueError("Model not loaded. Call load_model first.")
            
        # 执行检索
        results = self.rag_pipeline.retrieve(
            queries=[query],
            top_k=top_k,
            use_index=True
        )
        
        # 转换为RetrievalResult格式
        retrieval_results = []
        for result in results[0]:  # results是List[List[Dict]]格式
            retrieval_results.append(RetrievalResult(
                page_num=result['page_num'],
                page_index=result['page_index'],
                score=float(result['score']),
                source="nvidia_faiss",
                content_type="document"
            ))
            
        return retrieval_results

class MultiPathRetriever:
    """多路召回系统"""
    
    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self.metaclip_retriever = MetaCLIPRetriever(device=device)
        self.nvidia_retriever = NvidiaFAISSRetriever(device=device)
        self.images = None
        
    def load_models(self, 
                   metaclip_token: Optional[str] = None,
                   nvidia_model: str = "nvidia/llama-nemoretriever-colembed-3b-v1"):
        """加载所有模型"""
        logger.info("Loading all retrieval models...")
        
        # 加载MetaCLIP2
        self.metaclip_retriever.load_model(hf_token=metaclip_token)
        
        # 加载Nvidia模型
        self.nvidia_retriever.load_model(model_name=nvidia_model)
        
        logger.info("All models loaded successfully")
        
    def build_indices(self, 
                     images: List[Image.Image],
                     metaclip_index_path: Optional[str] = None,
                     nvidia_index_dir: Optional[str] = None):
        """构建所有索引"""
        logger.info("Building all indices...")
        
        self.images = images
        
        # 构建MetaCLIP索引
        self.metaclip_retriever.build_image_index(
            images=images,
            save_path=metaclip_index_path
        )
        
        # 构建Nvidia索引
        self.nvidia_retriever.build_document_index(
            images=images,
            save_dir=nvidia_index_dir
        )
        
        logger.info("All indices built successfully")
        
    def retrieve(self, 
                query: str, 
                top_k_per_path: int = 20,
                final_top_k: int = 10,
                fusion_method: str = "weighted_sum",
                metaclip_weight: float = 0.4,
                nvidia_weight: float = 0.6) -> List[FusedResult]:
        """多路召回并融合结果"""
        
        logger.info(f"Multi-path retrieval for query: {query}")
        
        # 路径1：MetaCLIP检索
        start_time = time.time()
        metaclip_results = self.metaclip_retriever.search(query, top_k_per_path)
        metaclip_time = time.time() - start_time
        logger.info(f"MetaCLIP retrieval: {len(metaclip_results)} results in {metaclip_time:.3f}s")
        
        # 路径2：Nvidia FAISS检索
        start_time = time.time()
        nvidia_results = self.nvidia_retriever.search(query, top_k_per_path)
        nvidia_time = time.time() - start_time
        logger.info(f"Nvidia FAISS retrieval: {len(nvidia_results)} results in {nvidia_time:.3f}s")
        
        # 融合结果
        fused_results = self._fuse_results(
            metaclip_results=metaclip_results,
            nvidia_results=nvidia_results,
            fusion_method=fusion_method,
            metaclip_weight=metaclip_weight,
            nvidia_weight=nvidia_weight,
            top_k=final_top_k
        )
        
        logger.info(f"Final fused results: {len(fused_results)} pages")
        return fused_results
        
    def _fuse_results(self,
                     metaclip_results: List[RetrievalResult],
                     nvidia_results: List[RetrievalResult],
                     fusion_method: str = "rrf",
                     metaclip_weight: float = 0.4,
                     nvidia_weight: float = 0.6,
                     top_k: int = 10) -> List[FusedResult]:
        """融合多路召回结果"""
        
        # 创建页面到分数的映射
        metaclip_scores = {r.page_index: r.score for r in metaclip_results}
        nvidia_scores = {r.page_index: r.score for r in nvidia_results}
        
        # 获取所有涉及的页面
        all_pages = set(metaclip_scores.keys()) | set(nvidia_scores.keys())
        
        fused_results = []
        
        for page_idx in all_pages:
            metaclip_score = metaclip_scores.get(page_idx, 0.0)
            nvidia_score = nvidia_scores.get(page_idx, 0.0)
            
            # 分数归一化（可选）
            if fusion_method == "weighted_sum":
                final_score = (metaclip_weight * metaclip_score + 
                             nvidia_weight * nvidia_score)
                             
            elif fusion_method == "max":
                final_score = max(metaclip_score, nvidia_score)
                
            elif fusion_method == "rrf":  # Reciprocal Rank Fusion
                # 计算排名
                metaclip_rank = self._get_rank(page_idx, metaclip_results)
                nvidia_rank = self._get_rank(page_idx, nvidia_results)
                
                # RRF公式：1/(k + rank)，k通常取60
                k = 60
                rrf_score = 0
                if metaclip_rank > 0:
                    rrf_score += 1.0 / (k + metaclip_rank)
                if nvidia_rank > 0:
                    rrf_score += 1.0 / (k + nvidia_rank)
                    
                final_score = rrf_score
                
            else:
                # 默认加权求和
                final_score = (metaclip_weight * metaclip_score + 
                             nvidia_weight * nvidia_score)
            
            fused_results.append(FusedResult(
                page_num=page_idx + 1,
                page_index=page_idx,
                final_score=final_score,
                metaclip_score=metaclip_score if metaclip_score > 0 else None,
                nvidia_score=nvidia_score if nvidia_score > 0 else None,
                fusion_method=fusion_method
            ))
        
        # 按最终分数排序并返回top_k
        fused_results.sort(key=lambda x: x.final_score, reverse=True)
        return fused_results[:top_k]
        
    def _get_rank(self, page_idx: int, results: List[RetrievalResult]) -> int:
        """获取页面在结果中的排名（1-based）"""
        for rank, result in enumerate(results, 1):
            if result.page_index == page_idx:
                return rank
        return 0  # 未找到
        
    def print_results(self, results: List[FusedResult], query: str):
        """打印融合结果"""
        print(f"\n{'='*80}")
        print(f"查询: {query}")
        print(f"{'='*80}")
        print(f"融合方法: {results[0].fusion_method if results else 'N/A'}")
        print(f"Top-{len(results)} 结果:")
        print("-" * 80)
        
        for rank, result in enumerate(results, 1):
            print(f"{rank:2d}. 第 {result.page_num:3d} 页")
            print(f"     融合分数: {result.final_score:.4f}")
            if result.metaclip_score is not None:
                print(f"     MetaCLIP:  {result.metaclip_score:.4f}")
            if result.nvidia_score is not None:
                print(f"     Nvidia:    {result.nvidia_score:.4f}")
            print()

def load_pdf_images(pdf_path: str, dpi: int = 200) -> List[Image.Image]:
    """加载PDF为图像列表"""
    logger.info(f"Loading PDF: {pdf_path}")
    images = convert_from_path(pdf_path, dpi=dpi)
    logger.info(f"PDF loaded: {len(images)} pages")
    return images

def main():
    """主函数示例"""
    
    # 配置
    DEVICE = "cuda:3"
    PDF_PATH = "contents/2024_Tencent_ESG.pdf"
    HF_TOKEN = os.getenv("HF_TOKEN")
    
    # 测试查询
    queries = [
        "2022年员工总数是多少？",
        "以2021年作为基准年，2023年温室气体范围3排放量是否达到了减少的目标？",
        "截至二零二四年每收入单位的温室气体排放总量是多少？"
    ]
    
    try:
        # 1. 加载PDF
        images = load_pdf_images(PDF_PATH)
        
        # 2. 初始化多路召回系统
        retriever = MultiPathRetriever(device=DEVICE)
        
        # 3. 加载模型
        retriever.load_models(metaclip_token=HF_TOKEN)
        
        # 4. 构建索引
        retriever.build_indices(
            images=images,
            metaclip_index_path="./indices/metaclip_index.faiss",
            nvidia_index_dir="./indices/nvidia_faiss"
        )
        
        # 5. 执行多路召回
        for query in queries:
            # 测试不同融合方法
            for fusion_method in ["weighted_sum", "max", "rrf"]:
                results = retriever.retrieve(
                    query=query,
                    top_k_per_path=20,
                    final_top_k=10,
                    fusion_method=fusion_method,
                    metaclip_weight=0.4,
                    nvidia_weight=0.6
                )
                
                print(f"\n融合方法: {fusion_method}")
                retriever.print_results(results, query)
                
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main()
