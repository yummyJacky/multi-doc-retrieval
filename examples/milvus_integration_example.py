"""
Milvus 云向量数据库集成示例
完整的端到端示例，展示如何将现有的 RAG Pipeline 迁移到 Milvus
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import numpy as np
from transformers import AutoModel
from pdf2image import convert_from_path
from typing import List, Dict, Tuple
from tqdm import tqdm

# 导入现有的 RAG Pipeline
import sys
sys.path.append('..')
from nvidia_rag_with_faiss import NvidiaRAGPipeline, GPUMemoryMonitor


class MilvusRAGPipeline(NvidiaRAGPipeline):
    """
    基于 Milvus 的 RAG Pipeline
    继承原有功能，替换 FAISS 为 Milvus
    """
    
    def __init__(
        self,
        retriever_model,
        milvus_host: str = "localhost",
        milvus_port: int = 19530,
        collection_name: str = "document_embeddings",
        device: int = 0
    ):
        """
        初始化 Milvus RAG Pipeline
        
        Args:
            retriever_model: Nvidia NemoRetriever 模型
            milvus_host: Milvus 服务器地址
            milvus_port: Milvus 端口
            collection_name: 集合名称
            device: GPU 设备 ID
        """
        # 初始化父类（不使用 FAISS）
        super().__init__(
            retriever_model=retriever_model,
            use_faiss=False,
            device=device
        )
        
        # 连接到 Milvus
        self._connect_milvus(milvus_host, milvus_port, collection_name)
    
    def _connect_milvus(self, host: str, port: int, collection_name: str):
        """连接到 Milvus 并创建集合"""
        try:
            from pymilvus import (
                connections, Collection, FieldSchema, 
                CollectionSchema, DataType, utility
            )
        except ImportError:
            raise ImportError(
                "请安装 pymilvus: pip install pymilvus\n"
                "或使用 Docker 运行 Milvus:\n"
                "docker run -d --name milvus_standalone -p 19530:19530 milvusdb/milvus:latest"
            )
        
        # 连接
        connections.connect(host=host, port=port)
        print(f"✓ 已连接到 Milvus: {host}:{port}")
        
        self.collection_name = collection_name
        
        # 创建或加载集合
        if utility.has_collection(collection_name):
            self.collection = Collection(collection_name)
            print(f"✓ 加载已存在的集合: {collection_name}")
        else:
            # 定义 schema
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="page_id", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="page_index", dtype=DataType.INT64),
                FieldSchema(name="token_index", dtype=DataType.INT64),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=3072),
            ]
            schema = CollectionSchema(fields=fields, description="Document token embeddings")
            
            # 创建集合
            self.collection = Collection(name=collection_name, schema=schema)
            print(f"✓ 创建新集合: {collection_name}")
            
            # 创建索引
            index_params = {
                "metric_type": "IP",  # 内积相似度
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
            self.collection.create_index(field_name="embedding", index_params=index_params)
            print("✓ 创建索引完成")
    
    def build_unified_index(self, save_dir=None):
        """
        构建统一的多文档索引（上传到 Milvus）
        
        Args:
            save_dir: 忽略此参数（Milvus 自动持久化）
        """
        if not self.is_multi_doc_mode or len(self.docid2embeddings) == 0:
            raise ValueError("请先调用 encode_documents() 编码多个文档")
        
        print(f"\n{'='*80}")
        print(f"构建统一多文档索引（上传到 Milvus）")
        print(f"{'='*80}")
        
        # 准备数据
        all_data = []
        
        for doc_id in sorted(self.docid2embeddings.keys()):
            embeddings = self.docid2embeddings[doc_id]  # (n_pages, n_tokens, dim)
            page_ids = self.docid2page_ids[doc_id]
            
            print(f"  - 处理文档: {doc_id} ({len(page_ids)} 页)")
            
            # 展平嵌入向量
            for page_idx, (page_emb, page_id) in enumerate(zip(embeddings, page_ids)):
                # page_emb: (n_tokens, dim)
                for token_idx, token_emb in enumerate(page_emb):
                    all_data.append({
                        "page_id": page_id,
                        "doc_id": doc_id,
                        "page_index": page_idx,
                        "token_index": token_idx,
                        "embedding": token_emb.tolist()
                    })
        
        print(f"\n✓ 准备完成：总共 {len(all_data)} 个 token 嵌入")
        
        # 批量插入
        batch_size = 1000
        print(f"\n批量插入到 Milvus (batch_size={batch_size})...")
        
        for i in tqdm(range(0, len(all_data), batch_size), desc="上传数据"):
            batch = all_data[i:i+batch_size]
            
            # 准备批次数据
            entities = [
                [d["page_id"] for d in batch],
                [d["doc_id"] for d in batch],
                [d["page_index"] for d in batch],
                [d["token_index"] for d in batch],
                [d["embedding"] for d in batch],
            ]
            
            # 插入
            self.collection.insert(entities)
        
        # 刷新以确保数据持久化
        self.collection.flush()
        
        print(f"\n✓ 上传完成！")
        print(f"  - 集合名称: {self.collection_name}")
        print(f"  - 总向量数: {self.collection.num_entities}")
        print(f"{'='*80}\n")
    
    def retrieve_multi_doc(
        self,
        queries: List[str],
        top_k: int = 10,
        single_page_per_doc: bool = False,
        batch_size: int = 8,
        nprobe: int = 10
    ) -> List[List[Dict]]:
        """
        多文档检索（使用 Milvus）
        
        Args:
            queries: 查询文本列表
            top_k: 返回 top-k 个页面
            single_page_per_doc: 是否每个文档只返回一页
            batch_size: 批处理大小
            nprobe: Milvus 搜索参数（探测的聚类数量）
            
        Returns:
            List[List[Dict]]: 每个查询对应一个结果列表
        """
        if not self.is_multi_doc_mode:
            raise ValueError("请先调用 encode_documents() 进入多文档模式")
        
        # 加载集合到内存
        self.collection.load()
        
        # 编码查询
        print(f"\n编码 {len(queries)} 个查询...")
        query_embeddings = self.retriever_model.forward_queries(
            queries, batch_size=batch_size
        )
        query_embeddings_cpu = query_embeddings.cpu().numpy()
        
        # 搜索参数
        search_params = {
            "metric_type": "IP",
            "params": {"nprobe": nprobe}
        }
        
        # 对每个查询进行检索
        all_results = []
        
        print(f"检索中...")
        for q_idx, query in enumerate(tqdm(queries, desc="查询进度")):
            query_emb = query_embeddings_cpu[q_idx]  # (n_tokens, dim)
            
            # 搜索（每个 query token 搜索）
            results = self.collection.search(
                data=query_emb.tolist(),
                anns_field="embedding",
                param=search_params,
                limit=top_k * 20,  # 多检索一些用于 MaxSim 聚合
                output_fields=["page_id", "doc_id", "page_index"]
            )
            
            # MaxSim 聚合：每个页面取最大分数
            page_scores = {}
            page_info = {}
            
            for hits in results:
                for hit in hits:
                    page_id = hit.entity.get("page_id")
                    doc_id = hit.entity.get("doc_id")
                    page_index = hit.entity.get("page_index")
                    score = hit.distance
                    
                    if page_id in page_scores:
                        page_scores[page_id] = max(page_scores[page_id], score)
                    else:
                        page_scores[page_id] = score
                        page_info[page_id] = {
                            "doc_id": doc_id,
                            "page_index": page_index
                        }
            
            # 应用检索策略
            if single_page_per_doc:
                # 每个文档只返回一页
                from nvidia_rag_with_faiss import get_top_k_pages_single_page_from_each_doc
                
                docid2scores = {}
                for page_id, score in page_scores.items():
                    doc_id = page_info[page_id]["doc_id"]
                    page_idx = page_info[page_id]["page_index"]
                    
                    if doc_id not in docid2scores:
                        docid2scores[doc_id] = []
                    docid2scores[doc_id].append(score)
                
                top_pages = get_top_k_pages_single_page_from_each_doc(docid2scores, k=top_k)
                
                # 格式化结果
                formatted_results = []
                for rank, (doc_id, page_idx, score) in enumerate(top_pages, 1):
                    page_id = f"{doc_id}_page_{page_idx}"
                    formatted_results.append({
                        "query": query,
                        "page_id": page_id,
                        "doc_id": doc_id,
                        "page_index": page_idx,
                        "page_num": page_idx + 1,
                        "score": float(score),
                        "rank": rank
                    })
            else:
                # 跨文档跨页面检索
                sorted_pages = sorted(page_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
                
                formatted_results = []
                for rank, (page_id, score) in enumerate(sorted_pages, 1):
                    info = page_info[page_id]
                    formatted_results.append({
                        "query": query,
                        "page_id": page_id,
                        "doc_id": info["doc_id"],
                        "page_index": info["page_index"],
                        "page_num": info["page_index"] + 1,
                        "score": float(score),
                        "rank": rank
                    })
            
            all_results.append(formatted_results)
        
        return all_results


def main():
    """完整示例"""
    print("\n" + "="*80)
    print("Milvus 云向量数据库集成示例")
    print("="*80)
    
    DEVICE = 9
    
    # 1. 加载 Retriever 模型
    print("\n1. 加载 Retriever 模型...")
    retriever_model = AutoModel.from_pretrained(
        'nvidia/llama-nemoretriever-colembed-3b-v1',
        device_map=f'cuda:{DEVICE}',
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        revision='50c36f4d5271c6851aa08bd26d69f6e7ca8b870c',
    ).eval()
    print("✓ 模型加载完成")
    
    # 2. 初始化 Milvus RAG Pipeline
    print("\n2. 初始化 Milvus RAG Pipeline...")
    rag_pipeline = MilvusRAGPipeline(
        retriever_model=retriever_model,
        milvus_host="localhost",  # 或云服务器地址
        milvus_port=19530,
        collection_name="multi_doc_demo",
        device=DEVICE
    )
    print("✓ Pipeline 初始化完成")
    
    # 3. 加载文档
    print("\n3. 加载文档...")
    documents = {
        "tencent_esg": "../contents/2024_Tencent_ESG.pdf",
        "archi_esg": "../contents/2024_architecture_ESG.pdf",
    }
    
    docid2images = {}
    for doc_id, pdf_path in documents.items():
        if os.path.exists(pdf_path):
            images = convert_from_path(pdf_path, dpi=200)
            docid2images[doc_id] = images[:10]  # 只取前10页作为演示
            print(f"  ✓ {doc_id}: {len(docid2images[doc_id])} 页")
    
    # 4. 编码文档
    print("\n4. 编码文档...")
    rag_pipeline.encode_documents(docid2images, batch_size=4)
    
    # 5. 构建索引（上传到 Milvus）
    print("\n5. 构建索引并上传到 Milvus...")
    rag_pipeline.build_unified_index()
    
    # 6. 检索测试
    print("\n6. 检索测试...")
    queries = [
        "公司的可持续发展战略是什么？",
        "温室气体排放情况如何？"
    ]
    
    results = rag_pipeline.retrieve_multi_doc(
        queries=queries,
        top_k=5,
        single_page_per_doc=False
    )
    
    # 显示结果
    print("\n" + "="*80)
    print("检索结果")
    print("="*80)
    
    for i, (query, query_results) in enumerate(zip(queries, results)):
        print(f"\n查询 {i+1}: {query}")
        print("-" * 80)
        
        for result in query_results:
            doc_id = result['doc_id']
            page_num = result['page_num']
            score = result['score']
            rank = result['rank']
            print(f"  {rank}. 文档: {doc_id}, 第 {page_num} 页, 分数: {score:.4f}")
    
    print("\n" + "="*80)
    print("✓ 示例完成！")
    print("="*80)
    
    print("\n提示:")
    print("- 索引已保存在 Milvus 中，下次可以直接使用")
    print("- 可以通过 Milvus 的 Web UI 查看数据: http://localhost:9091")
    print("- 支持增量更新：只需再次调用 build_unified_index() 添加新文档")


if __name__ == "__main__":
    main()
