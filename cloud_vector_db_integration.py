"""
云向量数据库集成示例
支持 Milvus, Pinecone, Weaviate 等
"""

import torch
import numpy as np
from typing import List, Dict, Optional
from PIL import Image


# ============================================================================
# Milvus 集成示例
# ============================================================================

class MilvusIndexManager:
    """Milvus 向量数据库管理器"""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 19530,
        collection_name: str = "document_embeddings",
        embedding_dim: int = 3072,
    ):
        """
        初始化 Milvus 连接
        
        Args:
            host: Milvus 服务器地址
            port: Milvus 端口
            collection_name: 集合名称
            embedding_dim: 嵌入维度
        """
        try:
            from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
        except ImportError:
            raise ImportError("请安装 pymilvus: pip install pymilvus")
        
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        
        # 连接到 Milvus
        connections.connect(host=host, port=port)
        print(f"✓ 已连接到 Milvus: {host}:{port}")
        
        # 创建或加载集合
        self._setup_collection()
    
    def _setup_collection(self):
        """创建或加载集合"""
        from pymilvus import Collection, FieldSchema, CollectionSchema, DataType, utility
        
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            print(f"✓ 加载已存在的集合: {self.collection_name}")
        else:
            # 定义 schema
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="page_id", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="page_index", dtype=DataType.INT64),
                FieldSchema(name="token_index", dtype=DataType.INT64),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
            ]
            schema = CollectionSchema(fields=fields, description="Document embeddings")
            
            # 创建集合
            self.collection = Collection(name=self.collection_name, schema=schema)
            print(f"✓ 创建新集合: {self.collection_name}")
            
            # 创建索引
            index_params = {
                "metric_type": "IP",  # 内积相似度
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
            self.collection.create_index(field_name="embedding", index_params=index_params)
            print("✓ 创建索引完成")
    
    def insert_embeddings(
        self,
        embeddings: np.ndarray,
        page_ids: List[str],
        batch_size: int = 1000
    ):
        """
        插入嵌入向量到 Milvus
        
        Args:
            embeddings: (n_pages, n_tokens, dim)
            page_ids: 页面ID列表
            batch_size: 批处理大小
        """
        from tqdm import tqdm
        
        print(f"\n插入嵌入向量到 Milvus...")
        
        # 展平嵌入向量
        all_data = []
        for page_idx, page_emb in enumerate(tqdm(embeddings, desc="准备数据")):
            page_id = page_ids[page_idx]
            # 解析 doc_id
            if "_page_" in page_id:
                doc_id = page_id.rsplit("_page_", 1)[0]
            else:
                doc_id = "default"
            
            # 每个 token 一条记录
            for token_idx, token_emb in enumerate(page_emb):
                all_data.append({
                    "page_id": page_id,
                    "doc_id": doc_id,
                    "page_index": page_idx,
                    "token_index": token_idx,
                    "embedding": token_emb.tolist()
                })
        
        # 批量插入
        print(f"批量插入 {len(all_data)} 条记录...")
        for i in tqdm(range(0, len(all_data), batch_size), desc="插入数据"):
            batch = all_data[i:i+batch_size]
            entities = [
                [d["page_id"] for d in batch],
                [d["doc_id"] for d in batch],
                [d["page_index"] for d in batch],
                [d["token_index"] for d in batch],
                [d["embedding"] for d in batch],
            ]
            self.collection.insert(entities)
        
        # 刷新
        self.collection.flush()
        print(f"✓ 插入完成，总共 {len(all_data)} 条记录")
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        nprobe: int = 10
    ) -> List[Dict]:
        """
        搜索相似向量
        
        Args:
            query_embedding: (n_tokens, dim)
            top_k: 返回 top-k 个页面
            nprobe: 搜索参数
            
        Returns:
            [(page_id, score), ...]
        """
        # 加载集合到内存
        self.collection.load()
        
        # 搜索参数
        search_params = {"metric_type": "IP", "params": {"nprobe": nprobe}}
        
        # 对每个 query token 搜索
        query_vectors = query_embedding.tolist()
        results = self.collection.search(
            data=query_vectors,
            anns_field="embedding",
            param=search_params,
            limit=top_k * 10,  # 多检索一些
            output_fields=["page_id", "doc_id", "page_index"]
        )
        
        # MaxSim 聚合
        page_scores = {}
        for hits in results:
            for hit in hits:
                page_id = hit.entity.get("page_id")
                score = hit.distance
                
                if page_id in page_scores:
                    page_scores[page_id] = max(page_scores[page_id], score)
                else:
                    page_scores[page_id] = score
        
        # 排序并返回
        sorted_pages = sorted(page_scores.items(), key=lambda x: x[1], reverse=True)
        return [{"page_id": pid, "score": score} for pid, score in sorted_pages[:top_k]]


# ============================================================================
# Pinecone 集成示例
# ============================================================================

class PineconeIndexManager:
    """Pinecone 向量数据库管理器（云服务）"""
    
    def __init__(
        self,
        api_key: str,
        environment: str,
        index_name: str = "document-embeddings",
        embedding_dim: int = 3072,
    ):
        """
        初始化 Pinecone
        
        Args:
            api_key: Pinecone API key
            environment: Pinecone 环境（如 "us-west1-gcp"）
            index_name: 索引名称
            embedding_dim: 嵌入维度
        """
        try:
            import pinecone
        except ImportError:
            raise ImportError("请安装 pinecone: pip install pinecone-client")
        
        # 初始化 Pinecone
        pinecone.init(api_key=api_key, environment=environment)
        
        # 创建或连接索引
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=index_name,
                dimension=embedding_dim,
                metric="dotproduct"  # 内积相似度
            )
            print(f"✓ 创建新索引: {index_name}")
        
        self.index = pinecone.Index(index_name)
        print(f"✓ 连接到 Pinecone 索引: {index_name}")
    
    def insert_embeddings(
        self,
        embeddings: np.ndarray,
        page_ids: List[str],
        batch_size: int = 100
    ):
        """插入嵌入向量到 Pinecone"""
        from tqdm import tqdm
        
        print(f"\n插入嵌入向量到 Pinecone...")
        
        # 准备数据
        vectors = []
        for page_idx, page_emb in enumerate(tqdm(embeddings, desc="准备数据")):
            page_id = page_ids[page_idx]
            
            for token_idx, token_emb in enumerate(page_emb):
                vector_id = f"{page_id}_token_{token_idx}"
                vectors.append((
                    vector_id,
                    token_emb.tolist(),
                    {"page_id": page_id, "page_index": page_idx, "token_index": token_idx}
                ))
        
        # 批量插入
        print(f"批量插入 {len(vectors)} 个向量...")
        for i in tqdm(range(0, len(vectors), batch_size), desc="插入数据"):
            batch = vectors[i:i+batch_size]
            self.index.upsert(vectors=batch)
        
        print(f"✓ 插入完成")
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10
    ) -> List[Dict]:
        """搜索相似向量"""
        # 对每个 query token 搜索
        all_results = []
        for token_emb in query_embedding:
            results = self.index.query(
                vector=token_emb.tolist(),
                top_k=top_k * 10,
                include_metadata=True
            )
            all_results.extend(results["matches"])
        
        # MaxSim 聚合
        page_scores = {}
        for match in all_results:
            page_id = match["metadata"]["page_id"]
            score = match["score"]
            
            if page_id in page_scores:
                page_scores[page_id] = max(page_scores[page_id], score)
            else:
                page_scores[page_id] = score
        
        # 排序并返回
        sorted_pages = sorted(page_scores.items(), key=lambda x: x[1], reverse=True)
        return [{"page_id": pid, "score": score} for pid, score in sorted_pages[:top_k]]


# ============================================================================
# Weaviate 集成示例
# ============================================================================

class WeaviateIndexManager:
    """Weaviate 向量数据库管理器"""
    
    def __init__(
        self,
        url: str = "http://localhost:8080",
        class_name: str = "DocumentEmbedding",
        embedding_dim: int = 3072,
    ):
        """
        初始化 Weaviate
        
        Args:
            url: Weaviate 服务器 URL
            class_name: 类名称
            embedding_dim: 嵌入维度
        """
        try:
            import weaviate
        except ImportError:
            raise ImportError("请安装 weaviate: pip install weaviate-client")
        
        self.client = weaviate.Client(url)
        self.class_name = class_name
        
        # 创建 schema
        self._setup_schema(embedding_dim)
    
    def _setup_schema(self, embedding_dim: int):
        """创建 schema"""
        schema = {
            "class": self.class_name,
            "vectorizer": "none",  # 我们提供自己的向量
            "properties": [
                {"name": "page_id", "dataType": ["string"]},
                {"name": "doc_id", "dataType": ["string"]},
                {"name": "page_index", "dataType": ["int"]},
                {"name": "token_index", "dataType": ["int"]},
            ]
        }
        
        # 检查是否已存在
        if not self.client.schema.exists(self.class_name):
            self.client.schema.create_class(schema)
            print(f"✓ 创建 schema: {self.class_name}")
        else:
            print(f"✓ 加载已存在的 schema: {self.class_name}")


# ============================================================================
# 使用示例
# ============================================================================

def example_milvus():
    """Milvus 使用示例"""
    from nvidia_rag_with_faiss import NvidiaRAGPipeline
    
    # 1. 编码文档（使用现有的 RAG Pipeline）
    # ... (省略编码步骤)
    
    # 2. 初始化 Milvus
    milvus_manager = MilvusIndexManager(
        host="localhost",  # 或云服务器地址
        port=19530,
        collection_name="multi_doc_embeddings",
        embedding_dim=3072
    )
    
    # 3. 插入嵌入向量
    # embeddings: (n_pages, n_tokens, dim)
    # page_ids: List[str]
    # milvus_manager.insert_embeddings(embeddings, page_ids)
    
    # 4. 搜索
    # query_embedding: (n_tokens, dim)
    # results = milvus_manager.search(query_embedding, top_k=10)
    
    print("✓ Milvus 集成完成")


def example_pinecone():
    """Pinecone 使用示例"""
    # 1. 初始化 Pinecone
    pinecone_manager = PineconeIndexManager(
        api_key="your-api-key",
        environment="us-west1-gcp",
        index_name="document-embeddings",
        embedding_dim=3072
    )
    
    # 2. 插入和搜索（同 Milvus）
    print("✓ Pinecone 集成完成")


if __name__ == "__main__":
    print("云向量数据库集成示例")
    print("=" * 80)
    print("\n支持的数据库:")
    print("1. Milvus - 开源，可自建或使用 Zilliz Cloud")
    print("2. Pinecone - 云服务，按使用量付费")
    print("3. Weaviate - 开源，可自建或使用云服务")
    print("4. Qdrant - 开源，高性能")
    print("5. ChromaDB - 轻量级，适合小规模")
