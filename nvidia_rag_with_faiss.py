"""
Nvidia NemoRetriever RAG Pipeline with FAISS Indexing
重构版本：添加向量索引支持，提升大规模文档检索性能
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
        embedding_dim: int = 3072,  # Nvidia NemoRetriever 的实际维度
        index_type: str = "flat",  # 'flat', 'ivfflat', 'ivfpq'
        nlist: int = 100,  # IVF聚类中心数量
        use_gpu: bool = False,
        gpu_id: int = 0
    ):
        """
        初始化FAISS索引
        
        Args:
            embedding_dim: 嵌入向量维度
            index_type: 索引类型
                - 'flat': 精确搜索，适合小规模(<10K)
                - 'ivfflat': 倒排索引，适合中等规模(10K-1M)
                - 'ivfpq': 乘积量化，适合大规模(>1M)
            nlist: IVF索引的聚类中心数量
            use_gpu: 是否使用GPU加速索引
            gpu_id: GPU设备ID
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.nlist = nlist
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        
        self.index = None
        self.page_id_map = []  # 存储 page_uid 映射
        self.is_trained = False
        
    def _create_index(self) -> faiss.Index:
        """创建FAISS索引"""
        if self.index_type == "flat":
            # 精确搜索索引
            index = faiss.IndexFlatIP(self.embedding_dim)  # 内积相似度
            
        elif self.index_type == "ivfflat":
            # IVF倒排索引
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, self.nlist)
            
        elif self.index_type == "ivfpq":
            # IVF + 乘积量化
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            m = 8  # 子向量数量
            nbits = 8  # 每个子向量的比特数
            index = faiss.IndexIVFPQ(quantizer, self.embedding_dim, self.nlist, m, nbits)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        # GPU加速
        if self.use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, self.gpu_id, index)
            print(f"✓ FAISS索引已移至GPU {self.gpu_id}")
        
        return index
    
    def build_index(
        self,
        embeddings: np.ndarray,
        page_ids: List[str],
        batch_size: int = 8
    ):
        """
        构建FAISS索引
        
        Args:
            embeddings: 嵌入向量 (n_pages, n_tokens, dim)
            page_ids: 页面ID列表 (n_pages,)
            batch_size: 批处理大小
        """
        print(f"\n{'='*80}")
        print(f"构建FAISS索引 (类型: {self.index_type})")
        print(f"{'='*80}")
        
        # 展平嵌入向量: (n_pages, n_tokens, dim) -> (n_total_tokens, dim)
        all_token_embeddings = []
        token_to_page_map = []
        
        print("展平嵌入向量...")
        for page_idx, page_emb in enumerate(tqdm(embeddings, desc="Processing pages")):
            # page_emb: (n_tokens, dim)
            n_tokens = page_emb.shape[0]
            all_token_embeddings.append(page_emb)
            token_to_page_map.extend([page_ids[page_idx]] * n_tokens)
        
        # 合并所有token嵌入
        all_token_embeddings = np.vstack(all_token_embeddings).astype(np.float32)
        print(f"✓ 总共 {len(all_token_embeddings)} 个token嵌入")
        
        # 归一化（用于内积相似度）
        faiss.normalize_L2(all_token_embeddings)
        
        # 创建索引
        self.index = self._create_index()
        self.page_id_map = token_to_page_map
        
        # 训练索引（IVF类型需要）
        if self.index_type in ["ivfflat", "ivfpq"]:
            print(f"训练索引 (nlist={self.nlist})...")
            # 使用部分数据训练（最多100K样本）
            train_size = min(100000, len(all_token_embeddings))
            train_data = all_token_embeddings[:train_size]
            self.index.train(train_data)
            self.is_trained = True
            print("✓ 索引训练完成")
        
        # 添加向量到索引
        print("添加向量到索引...")
        n_batches = (len(all_token_embeddings) + batch_size - 1) // batch_size
        
        for i in tqdm(range(n_batches), desc="Adding vectors"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(all_token_embeddings))
            batch = all_token_embeddings[start_idx:end_idx]
            self.index.add(batch)
        
        print(f"✓ 索引构建完成！总共 {self.index.ntotal} 个向量")
        print(f"{'='*80}\n")
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        nprobe: int = 10
    ) -> List[Tuple[str, float]]:
        """
        使用MaxSim策略搜索最相关的页面
        
        Args:
            query_embedding: 查询嵌入 (n_query_tokens, dim)
            top_k: 返回top-k个页面
            nprobe: IVF索引的探测聚类数量
            
        Returns:
            [(page_id, score), ...] 按分数降序排列
        """
        if self.index is None:
            raise ValueError("索引未构建，请先调用 build_index()")
        
        # 设置nprobe（仅对IVF索引有效）
        if self.index_type in ["ivfflat", "ivfpq"]:
            if hasattr(self.index, 'nprobe'):
                self.index.nprobe = nprobe
        
        # 归一化查询向量
        query_embedding = query_embedding.astype(np.float32)
        faiss.normalize_L2(query_embedding)
        print(f"query_embedding.shape: {query_embedding.shape}")
        # 对每个查询token搜索最近邻
        k_per_token = top_k * 10  # 每个token搜索更多候选
        D, I = self.index.search(query_embedding, k_per_token)
        
        # MaxSim聚合：对每个页面，取所有查询token的最大相似度之和
        page_scores = {}
        
        for q_idx in range(len(query_embedding)):
            # 当前查询token的最近邻
            token_scores = {}
            
            for nn_idx in range(k_per_token):
                token_idx = I[q_idx, nn_idx]
                if token_idx < 0 or token_idx >= len(self.page_id_map):
                    continue
                
                page_id = self.page_id_map[token_idx]
                score = D[q_idx, nn_idx]
                
                # MaxSim: 每个页面保留最大分数
                if page_id not in token_scores:
                    token_scores[page_id] = score
                else:
                    token_scores[page_id] = max(token_scores[page_id], score)
            
            # 累加所有查询token的分数
            for page_id, score in token_scores.items():
                if page_id in page_scores:
                    page_scores[page_id] += score
                else:
                    page_scores[page_id] = score
        
        # 排序并返回top-k
        sorted_pages = sorted(page_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_pages[:top_k]
    
    def save(self, save_dir: str):
        """保存索引到磁盘"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存FAISS索引
        if self.use_gpu:
            # GPU索引需要先转回CPU
            # 对于大规模索引，直接转换可能导致内存溢出
            # 解决方案：重新创建CPU索引并重新训练/添加
            print("⚠️ GPU索引较大，正在转换为CPU索引...")
            try:
                cpu_index = faiss.index_gpu_to_cpu(self.index)
                faiss.write_index(cpu_index, str(save_dir / "faiss.index"))
            except RuntimeError as e:
                print(f"⚠️ GPU->CPU转换失败: {e}")
                print("建议：使用 use_gpu=False 创建CPU索引，或不保存索引")
                # 只保存配置和映射，不保存索引
                print("跳过索引保存，仅保存配置和映射")
        else:
            faiss.write_index(self.index, str(save_dir / "faiss.index"))
        
        # 保存page_id映射
        with open(save_dir / "page_id_map.pkl", "wb") as f:
            pickle.dump(self.page_id_map, f)
        
        # 保存配置
        config = {
            "embedding_dim": self.embedding_dim,
            "index_type": self.index_type,
            "nlist": self.nlist,
        }
        with open(save_dir / "config.pkl", "wb") as f:
            pickle.dump(config, f)
        
        print(f"✓ 配置和映射已保存到 {save_dir}")
    
    def load(self, load_dir: str):
        """从磁盘加载索引"""
        load_dir = Path(load_dir)
        
        # 加载配置
        with open(load_dir / "config.pkl", "rb") as f:
            config = pickle.load(f)
        
        self.embedding_dim = config["embedding_dim"]
        self.index_type = config["index_type"]
        self.nlist = config["nlist"]
        
        # 加载FAISS索引
        self.index = faiss.read_index(str(load_dir / "faiss.index"))
        
        # GPU加速
        if self.use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, self.gpu_id, self.index)
        
        # 加载page_id映射
        with open(load_dir / "page_id_map.pkl", "rb") as f:
            self.page_id_map = pickle.load(f)
        
        print(f"✓ 索引已从 {load_dir} 加载")
        print(f"  - 索引类型: {self.index_type}")
        print(f"  - 向量数量: {self.index.ntotal}")
        print(f"  - 页面数量: {len(set(self.page_id_map))}")


class NvidiaRAGPipeline:
    """Nvidia NemoRetriever RAG Pipeline with FAISS"""
    
    def __init__(
        self,
        retriever_model,
        reranker=None,
        vqa_model=None,
        use_faiss: bool = True,
        faiss_config: Optional[Dict] = None,
        device: int = 0
    ):
        """
        初始化RAG Pipeline
        
        Args:
            retriever_model: Nvidia NemoRetriever模型
            reranker: 重排序模型（可选）
            vqa_model: VQA模型（可选）
            use_faiss: 是否使用FAISS索引
            faiss_config: FAISS配置
            device: GPU设备ID
        """
        self.retriever_model = retriever_model
        self.reranker = reranker
        self.vqa_model = vqa_model
        self.use_faiss = use_faiss
        self.device = device
        
        self.memory_monitor = GPUMemoryMonitor(device)
        
        # FAISS索引管理器
        if use_faiss:
            faiss_config = faiss_config or {}
            self.faiss_manager = FAISSIndexManager(
                embedding_dim=faiss_config.get("embedding_dim", 3072),  # Nvidia NemoRetriever 默认维度
                index_type=faiss_config.get("index_type", "ivfflat"),
                nlist=faiss_config.get("nlist", 100),
                use_gpu=faiss_config.get("use_gpu", False),
                gpu_id=device
            )
        else:
            self.faiss_manager = None
        
        # 单文档模式
        self.passage_embeddings = None
        self.page_ids = None
        
        # 多文档模式
        self.docid2embeddings: Dict[str, torch.Tensor] = {}  # {doc_id: embeddings [n_pages, n_tokens, dim]}
        self.docid2page_ids: Dict[str, List[str]] = {}  # {doc_id: [page_ids]}
        self.is_multi_doc_mode = False
        
        # Token级索引（M3DocRAG风格）
        self.all_token_embeddings = None  # 扁平化的所有token嵌入 [total_tokens, dim]
        self.token2pageuid = None  # token到页面的映射 List[str]
    
    def encode_passages(
        self,
        images: List[Image.Image],
        batch_size: int = 8,
        page_id_prefix: str = "page"
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        编码单个文档的图片
        
        Args:
            images: PIL图片列表
            batch_size: 批处理大小
            page_id_prefix: 页面ID前缀
            
        Returns:
            (embeddings, page_ids)
        """
        print(f"\n编码 {len(images)} 个文档页面...")
        self.memory_monitor.print_memory("Before passage encoding")
        
        # 生成页面ID
        page_ids = [f"{page_id_prefix}_{i}" for i in range(len(images))]
        
        # 编码
        passage_embeddings = self.retriever_model.forward_passages(
            images, batch_size=batch_size
        )
        
        self.memory_monitor.print_memory("After passage encoding")
        
        # 移到CPU节省显存
        passage_embeddings_cpu = passage_embeddings.cpu()
        del passage_embeddings
        GPUMemoryMonitor.clear_memory()
        
        self.passage_embeddings = passage_embeddings_cpu
        self.page_ids = page_ids
        
        print(f"✓ 文档编码完成")
        return passage_embeddings_cpu, page_ids
    
    def encode_documents(
        self,
        docid2images: Dict[str, List[Image.Image]],
        batch_size: int = 8
    ) -> Dict[str, torch.Tensor]:
        """
        编码多个文档（Multi-Document模式）
        
        Args:
            docid2images: {doc_id: [images]} 文档ID到图片列表的映射
            batch_size: 批处理大小
            
        Returns:
            docid2embeddings: {doc_id: embeddings}
        """
        print(f"\n{'='*80}")
        print(f"多文档编码模式：共 {len(docid2images)} 个文档")
        print(f"{'='*80}")
        
        self.is_multi_doc_mode = True
        self.docid2embeddings = {}
        self.docid2page_ids = {}
        
        for doc_id, images in docid2images.items():
            print(f"\n处理文档: {doc_id} ({len(images)} 页)")
            
            # 编码当前文档
            embeddings, page_ids = self.encode_passages(
                images=images,
                batch_size=batch_size,
                page_id_prefix=f"{doc_id}_page"
            )
            
            self.docid2embeddings[doc_id] = embeddings
            self.docid2page_ids[doc_id] = page_ids
            
            print(f"✓ 文档 {doc_id} 编码完成")
        
        print(f"\n{'='*80}")
        print(f"✓ 所有文档编码完成")
        print(f"{'='*80}\n")
        
        return self.docid2embeddings
    
    def build_index(self, save_dir: Optional[str] = None):
        """构建FAISS索引（单文档模式）"""
        if not self.use_faiss:
            print("⚠️ FAISS索引未启用")
            return
        
        if self.passage_embeddings is None:
            raise ValueError("请先调用 encode_passages() 编码文档")
        
        # 转换为numpy数组
        embeddings_np = self.passage_embeddings.numpy()
        
        # 构建索引
        self.faiss_manager.build_index(embeddings_np, self.page_ids)
        
        # 保存索引
        if save_dir:
            self.faiss_manager.save(save_dir)
    
    def build_unified_index(self, save_dir: Optional[str] = None):
        """
        构建统一的多文档Token级索引（M3DocRAG风格）
        
        关键改进：
        1. 扁平化所有token而不是页面
        2. 维护token到页面的映射关系
        3. 保留Late Interaction的细粒度匹配能力
        """
        if not self.use_faiss:
            print("⚠️ FAISS索引未启用")
            return
        
        if not self.is_multi_doc_mode or len(self.docid2embeddings) == 0:
            raise ValueError("请先调用 encode_documents() 编码多个文档")
        
        print(f"\n{'='*80}")
        print(f"构建Token级多文档索引（M3DocRAG风格）")
        print(f"{'='*80}")
        
        # Token级扁平化
        all_token_embeddings = []
        token2pageuid = []
        
        total_pages = 0
        total_tokens = 0
        
        for doc_id in sorted(self.docid2embeddings.keys()):
            doc_embs = self.docid2embeddings[doc_id]  # [n_pages, n_tokens, dim]
            n_pages = doc_embs.shape[0]
            
            print(f"  - {doc_id}: {n_pages} 页")
            
            # 遍历每一页
            for page_idx in range(n_pages):
                page_emb = doc_embs[page_idx]  # [n_tokens, dim]
                n_tokens = page_emb.shape[0]
                
                # 添加页面的所有token
                all_token_embeddings.append(page_emb)
                
                # 创建页面UID并记录映射
                page_uid = f"{doc_id}_page{page_idx}"
                token2pageuid.extend([page_uid] * n_tokens)
                
                total_tokens += n_tokens
            
            total_pages += n_pages
        
        print(f"\n✓ Token扁平化完成：")
        print(f"  - 总页面数: {total_pages}")
        print(f"  - 总Token数: {total_tokens}")
        
        # 合并所有token嵌入
        all_token_embeddings = torch.cat(all_token_embeddings, dim=0)  # [total_tokens, dim]
        print(f"  - 嵌入形状: {all_token_embeddings.shape}")
        
        # 保存到实例变量
        self.all_token_embeddings = all_token_embeddings.cpu().numpy().astype(np.float32)
        self.token2pageuid = token2pageuid
        
        # 构建FAISS索引
        print(f"\n构建FAISS索引...")
        self.faiss_manager.index = None  # 清空旧索引
        
        # 重新初始化索引
        embedding_dim = self.all_token_embeddings.shape[1]
        quantizer = faiss.IndexFlatIP(embedding_dim)
        
        if self.faiss_manager.index_type == "flatip":
            index = quantizer
        elif self.faiss_manager.index_type == "ivfflat":
            index = faiss.IndexIVFFlat(
                quantizer, 
                embedding_dim, 
                self.faiss_manager.nlist
            )
        else:
            raise ValueError(f"不支持的索引类型: {self.faiss_manager.index_type}")
        
        # 训练和添加数据
        if self.faiss_manager.index_type != "flatip":
            print(f"  训练索引...")
            index.train(self.all_token_embeddings)
        
        print(f"  添加向量...")
        index.add(self.all_token_embeddings)
        
        self.faiss_manager.index = index
        print(f"✓ 索引构建完成")
        
        # 保存索引和映射
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # 保存FAISS索引
            index_path = save_path / "index.bin"
            faiss.write_index(index, str(index_path))
            print(f"  - 索引已保存: {index_path}")
            
            # 保存token2pageuid映射
            mapping_path = save_path / "token2pageuid.pkl"
            with open(mapping_path, 'wb') as f:
                pickle.dump(self.token2pageuid, f)
            print(f"  - 映射已保存: {mapping_path}")

            # all_token_embeddings可能很大，所以只在save_dir存在时保存
            embeddings_path = save_path / "all_token_embeddings.npy"
            np.save(embeddings_path, self.all_token_embeddings)
            print(f"  - Token嵌入已保存: {embeddings_path}")
        
        print(f"{'='*80}\n")
    
    def load_index(self, load_dir: str):
        """加载已保存的索引和token映射"""
        if not self.use_faiss:
            print("⚠️ FAISS索引未启用")
            return
        
        load_path = Path(load_dir)
        
        # 加载FAISS索引
        index_path = load_path / "index.bin"
        if index_path.exists():
            self.faiss_manager.index = faiss.read_index(str(index_path))
            print(f"✓ 索引已加载: {index_path}")
        else:
            raise FileNotFoundError(f"索引文件不存在: {index_path}")
        
        # 加载token2pageuid映射
        mapping_path = load_path / "token2pageuid.pkl"
        if mapping_path.exists():
            with open(mapping_path, 'rb') as f:
                self.token2pageuid = pickle.load(f)
            print(f"✓ Token映射已加载: {mapping_path} ({len(self.token2pageuid)} tokens)")
        else:
            print(f"⚠️ 未找到token映射文件: {mapping_path}")
            self.token2pageuid = None
        
        # 尝试加载all_token_embeddings（用于精确分数计算）
        # 注意：这个文件可能很大，如果不存在也可以工作（但精度可能略低）
        embeddings_path = load_path / "all_token_embeddings.npy"
        if embeddings_path.exists():
            self.all_token_embeddings = np.load(embeddings_path)
            print(f"✓ Token嵌入已加载: {embeddings_path}")
        else:
            print(f"⚠️ 未找到token嵌入文件（将使用FAISS近似分数）")
    
    def retrieve(
        self,
        queries: List[str],
        top_k: int = 10,
        use_index: bool = True,
        batch_size: int = 8
    ):
        """
        检索相关页面（支持单个或多个查询）
        
        Args:
            queries: 查询文本列表，例如 ["问题1"] 或 ["问题1", "问题2", "问题3"]
            top_k: 返回top-k个页面
            use_index: 是否使用FAISS索引
            batch_size: 批处理大小
            
        Returns:
            List[List[Dict]]: 每个查询对应一个结果列表
            例如: [[query1_results], [query2_results], ...]
        """
        # 确保 queries 是列表
        if isinstance(queries, str):
            queries = [queries]
        
        # 批量编码查询
        query_embeddings = self.retriever_model.forward_queries(
            queries, batch_size=batch_size
        )
        query_embeddings_cpu = query_embeddings.cpu()
        
        # 对每个查询进行检索
        all_results = []
        
        for q_idx, single_query in enumerate(queries):
            query_embedding = query_embeddings_cpu[q_idx:q_idx+1]  # 保持2D形状
            
            if use_index and self.use_faiss and self.faiss_manager.index is not None:
                # 使用FAISS索引检索
                query_np = query_embedding.numpy()[0]  # (n_tokens, dim)
                results = self.faiss_manager.search(query_np, top_k=top_k)
                
                # 格式化结果
                formatted_results = []
                for page_id, score in results:
                    page_idx = int(page_id.split('_')[-1])
                    formatted_results.append({
                        "query": single_query,
                        "page_id": page_id,
                        "page_index": page_idx,
                        "page_num": page_idx + 1,
                        "score": float(score),
                    })
                
            else:
                # 直接计算相似度（无索引）
                scores = self.retriever_model.get_scores(
                    query_embedding,
                    self.passage_embeddings
                )
                
                # 获取top-k
                query_scores = scores[0]
                top_k_indices = query_scores.argsort(descending=True)[:top_k]
                
                formatted_results = []
                for rank, idx in enumerate(top_k_indices):
                    page_idx = idx.item()
                    formatted_results.append({
                        "query": single_query,
                        "page_id": self.page_ids[page_idx],
                        "page_index": page_idx,
                        "page_num": page_idx + 1,
                        "score": query_scores[idx].item(),
                        "rank": rank + 1
                    })
            
            all_results.append(formatted_results)
        
        return all_results
    
    def retrieve_multi_doc(
        self,
        queries: List[str],
        top_k: int = 10,
        single_page_per_doc: bool = False,
        batch_size: int = 8
    ) -> List[List[Dict]]:
        """
        多文档检索（M3DocRAG风格 - Token级MaxSim聚合）
        
        Args:
            queries: 查询文本列表
            top_k: 返回top-k个页面
            single_page_per_doc: 是否每个文档只返回一页（保证文档多样性）
            batch_size: 批处理大小
            
        Returns:
            List[List[Dict]]: 每个查询对应一个结果列表
            
        实现要点：
        1. 对每个查询token找到最近的文档token
        2. 使用MaxSim聚合：每个查询token对每个页面只保留最高分
        3. 对所有查询token的MaxSim分数求和
        4. 应用检索策略（跨页面或单页）
        """
        if not self.is_multi_doc_mode:
            raise ValueError("请先调用 encode_documents() 进入多文档模式")
        
        if self.faiss_manager.index is None:
            raise ValueError("请先调用 build_unified_index() 构建索引")
        
        if self.token2pageuid is None:
            raise ValueError("Token映射未加载，请确保索引已正确构建")
        
        # 确保queries是列表
        if isinstance(queries, str):
            queries = [queries]
        
        print(f"\n{'='*80}")
        print(f"多文档检索 - Token级MaxSim聚合")
        print(f"{'='*80}")
        print(f"查询数量: {len(queries)}")
        print(f"检索模式: {'每文档单页' if single_page_per_doc else '跨文档跨页面'}")
        
        # 编码查询
        query_embeddings = self.retriever_model.forward_queries(
            queries, batch_size=batch_size
        )
        query_embeddings_cpu = query_embeddings.cpu().numpy().astype(np.float32)
        
        all_results = []
        
        # 对每个查询进行检索
        for q_idx, query in enumerate(tqdm(queries, desc="检索查询")):
            query_emb = query_embeddings_cpu[q_idx]  # [n_query_tokens, dim]
            n_query_tokens = query_emb.shape[0]
            
            # FAISS最近邻搜索
            # 对每个查询token找k个最近的文档token
            k_nn = top_k * 10  # 多找一些候选
            D, I = self.faiss_manager.index.search(query_emb, k_nn)
            
            # MaxSim聚合
            final_page2scores = {}
            
            # 遍历每个查询token
            for token_idx in range(n_query_tokens):
                current_query_token_page2scores = {}
                
                # 遍历该查询token的k个最近邻
                for nn_idx in range(k_nn):
                    found_token_idx = I[token_idx, nn_idx]
                    
                    # 获取该token所属的页面
                    page_uid = self.token2pageuid[found_token_idx]
                    
                    # 计算精确分数
                    if self.all_token_embeddings is not None:
                        doc_token_emb = self.all_token_embeddings[found_token_idx]
                        score = (query_emb[token_idx] * doc_token_emb).sum()
                    else:
                        # 如果没有加载token嵌入，使用FAISS返回的距离
                        score = D[token_idx, nn_idx]
                    
                    # MaxSim: 每个查询token对每个页面只保留最高分
                    if page_uid not in current_query_token_page2scores:
                        current_query_token_page2scores[page_uid] = score
                    else:
                        current_query_token_page2scores[page_uid] = max(
                            current_query_token_page2scores[page_uid], score
                        )
                
                # 累加所有查询token的MaxSim分数
                for page_uid, score in current_query_token_page2scores.items():
                    if page_uid in final_page2scores:
                        final_page2scores[page_uid] += score
                    else:
                        final_page2scores[page_uid] = score
            
            # 排序页面
            sorted_pages = sorted(
                final_page2scores.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            # 解析页面UID
            parsed_results = []
            for page_uid, score in sorted_pages:
                # page_uid格式: "doc_id_page{page_idx}"
                parts = page_uid.rsplit('_page', 1)
                if len(parts) == 2:
                    doc_id = parts[0]
                    page_idx = int(parts[1])
                    parsed_results.append((doc_id, page_idx, float(score)))
            
            # 应用检索策略
            if single_page_per_doc:
                # 每个文档只保留最高分的一页
                seen_docs = set()
                top_pages = []
                for doc_id, page_idx, score in parsed_results:
                    if doc_id not in seen_docs:
                        seen_docs.add(doc_id)
                        top_pages.append((doc_id, page_idx, score))
                        if len(top_pages) >= top_k:
                            break
            else:
                # 跨文档跨页面，直接取top-k
                top_pages = parsed_results[:top_k]
            
            # 格式化结果
            formatted_results = []
            for rank, (doc_id, page_idx, score) in enumerate(top_pages, 1):
                page_uid = f"{doc_id}_page{page_idx}"
                formatted_results.append({
                    'query': query,
                    'doc_id': doc_id,
                    'page_idx': page_idx,
                    'page_num': page_idx + 1,
                    'page_id': page_uid,
                    'score': float(score),
                    'rank': rank
                })
            
            all_results.append(formatted_results)
        
        print(f"✓ 检索完成")
        print(f"{'='*80}\n")
        
        return all_results


class ImageReranker:
    """图片重排序器（使用 MonoVLM）"""
    
    def __init__(self, model_name: str = "monovlm", device: str = "cuda:0", use_fast: bool = True):
        """
        初始化 Reranker
        
        Args:
            model_name: 模型名称，默认 "monovlm"
            device: 设备，例如 "cuda:0"
            use_fast: 是否使用快速处理器
        """
        try:
            from rerankers import Reranker
        except ImportError:
            raise ImportError("请安装 rerankers 库: pip install rerankers")
        
        self.device = device
        print(f"Loading {model_name} reranker on {device}...")
        self.ranker = Reranker(model_name, use_fast=use_fast, device=device)
        print("✓ Reranker loaded successfully")
    
    def rerank(
        self,
        query: str,
        images: List[Image.Image],
        top_k: int = 5
    ) -> List[Dict]:
        """
        对图片进行重排序
        
        Args:
            query: 查询文本
            images: PIL 图片列表
            top_k: 返回前 k 个结果
            
        Returns:
            重排序后的结果列表，每个元素包含 {rank, doc_id, score}
        """
        # Base64转换函数
        import base64
        from io import BytesIO
        # 转换图片为 base64
        base64_images = []
        for img in images:
            buffer = BytesIO()
            img.save(buffer, format="JPEG")
            buffer.seek(0)
            
            img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            base64_images.append(img_base64)
        
        # 执行重排序
        rerank_results = self.ranker.rank(query, base64_images)
        
        # 提取 top-k 结果
        results = []
        for i, doc in enumerate(rerank_results.top_k(top_k)):
            results.append({
                'rank': i + 1,
                'doc_id': doc.doc_id,
                'score': doc.score
            })
        
        return results
    
    def rerank_batch(
        self,
        queries: List[str],
        all_images_list: List[List[Image.Image]],
        top_k: int = 5
    ) -> List[List[Dict]]:
        """
        批量重排序
        
        Args:
            queries: 查询列表
            all_images_list: 每个查询对应的图片列表
            top_k: 每个查询返回前 k 个结果
            
        Returns:
            每个查询的重排序结果
        """
        all_rerank_results = []
        
        for query, images in zip(queries, all_images_list):
            rerank_results = self.rerank(query, images, top_k=top_k)
            all_rerank_results.append(rerank_results)
        
        return all_rerank_results


class VQAModel:
    """多模态问答模型（支持 API 调用）"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model_name: str = "gpt-4o",
        use_api: bool = True
    ):
        """
        初始化 VQA 模型
        
        Args:
            api_key: API 密钥
            api_base: API 基础 URL
            model_name: 模型名称
            use_api: 是否使用 API（True）或本地模型（False）
        """
        self.use_api = use_api
        self.model_name = model_name
        
        if use_api:
            # 使用 API
            self.api_key = api_key or os.getenv("ARK_API_KEY")
            self.api_base = api_base or os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
            
            # 初始化 OpenAI 客户端
            self.client = OpenAI(
                base_url=self.api_base,
                api_key=self.api_key
            )
            
            print(f"✓ VQA Model initialized with API: {model_name}")
        else:
            # 使用本地模型（例如 Qwen2-VL）
            raise NotImplementedError("本地模型加载功能待实现，建议使用 API 模式避免显存不足")
    
    def answer_question(
        self,
        query: str,
        image: Image.Image,
        max_tokens: int = 512
    ) -> str:
        """
        基于单张图片回答问题
        
        Args:
            query: 问题
            image: PIL 图片
            max_tokens: 最大生成 token 数
            
        Returns:
            答案文本
        """
        if self.use_api:
            return self._answer_with_api(query, [image], max_tokens)
        else:
            return self._answer_with_local_model(query, [image], max_tokens)
    
    def answer_with_multiple_images(
        self,
        query: str,
        images: List[Image.Image],
        max_tokens: int = 512
    ) -> str:
        """
        基于多张图片综合回答问题（推荐用于 reranking 后的 top-k 结果）
        
        Args:
            query: 问题
            images: PIL 图片列表（例如 reranking top-5）
            max_tokens: 最大生成 token 数
            
        Returns:
            答案文本
        """
        if self.use_api:
            return self._answer_with_api(query, images, max_tokens)
        else:
            return self._answer_with_local_model(query, images, max_tokens)
    
    def _answer_with_api(self, query: str, images: List[Image.Image], max_tokens: int) -> str:
        """使用 API 回答问题（支持多图片）"""
        import base64
        from io import BytesIO
        
        # 构建 content 列表
        content = []
        
        # 添加问题文本
        if len(images) > 1:
            # 多图片时，添加提示语
            content.append({
                "type": "text",
                "text": f"请综合分析以下 {len(images)} 张图片，回答问题：{query}。若图片中没有符合关于query的内容，请回答不存在相关内容，不要编造。"
            })
        else:
            content.append({
                "type": "text",
                "text": query
            })
        
        # 添加所有图片
        for idx, image in enumerate(images):
            # 转换图片为 base64
            buffer = BytesIO()
            image.save(buffer, format="JPEG")
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_base64}"
                }
            })
        
        # 使用 OpenAI 客户端调用 API
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"API 请求失败: {str(e)}")
    
    def _answer_with_local_model(self, query: str, images: List[Image.Image], max_tokens: int) -> str:
        """使用本地模型回答问题（待实现）"""
        raise NotImplementedError("本地模型功能待实现")
    
    def answer_batch(
        self,
        queries: List[str],
        images: List[Image.Image],
        max_tokens: int = 512
    ) -> List[str]:
        """
        批量回答问题（每个查询对应单张图片）
        
        Args:
            queries: 问题列表
            images: 图片列表
            max_tokens: 最大生成 token 数
            
        Returns:
            答案列表
        """
        answers = []
        
        for query, image in tqdm(zip(queries, images), total=len(queries), desc="VQA Processing"):
            try:
                answer = self.answer_question(query, image, max_tokens)
                answers.append(answer)
            except Exception as e:
                print(f"Error processing query '{query}': {e}")
                answers.append(f"Error: {str(e)}")
        
        return answers
    
    def answer_batch_with_multiple_images(
        self,
        queries: List[str],
        all_images_list: List[List[Image.Image]],
        max_tokens: int = 512
    ) -> List[str]:
        """
        批量回答问题（每个查询对应多张图片，用于综合分析）
        
        Args:
            queries: 问题列表
            all_images_list: 每个查询对应的图片列表
                例如: [[img1_1, img1_2, img1_3], [img2_1, img2_2], ...]
            max_tokens: 最大生成 token 数
            
        Returns:
            答案列表
        """
        answers = []
        
        for query, images in tqdm(zip(queries, all_images_list), total=len(queries), desc="VQA Processing (Multi-Image)"):
            try:
                answer = self.answer_with_multiple_images(query, images, max_tokens)
                answers.append(answer)
            except Exception as e:
                print(f"Error processing query '{query}': {e}")
                answers.append(f"Error: {str(e)}")
        
        return answers

