"""
测试多文档检索功能
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from transformers import AutoModel
from pdf2image import convert_from_path
from nvidia_rag_with_faiss import (
    NvidiaRAGPipeline,
    get_top_k_pages,
    get_top_k_pages_single_page_from_each_doc
)

def test_utility_functions():
    """测试工具函数"""
    print("\n" + "="*80)
    print("测试工具函数")
    print("="*80)
    
    # 测试数据
    docid2scores = {
        "doc1": [10, 50, 30],
        "doc2": [40, 20, 60],
        "doc3": [70, 90]
    }
    
    # 测试 get_top_k_pages
    print("\n1. get_top_k_pages (跨文档跨页面):")
    top_pages = get_top_k_pages(docid2scores, k=3)
    for doc_id, page_idx, score in top_pages:
        print(f"   {doc_id}, 页 {page_idx}, 分数: {score}")
    
    # 测试 get_top_k_pages_single_page_from_each_doc
    print("\n2. get_top_k_pages_single_page_from_each_doc (每文档单页):")
    top_pages_single = get_top_k_pages_single_page_from_each_doc(docid2scores, k=2)
    for doc_id, page_idx, score in top_pages_single:
        print(f"   {doc_id}, 页 {page_idx}, 分数: {score}")
    
    print("\n✓ 工具函数测试通过")

def test_multi_doc_pipeline():
    """测试多文档检索Pipeline"""
    print("\n" + "="*80)
    print("测试多文档检索Pipeline")
    print("="*80)
    
    DEVICE = 9
    
    # 1. 加载模型
    print("\n1. 加载Retriever模型...")
    retriever_model = AutoModel.from_pretrained(
        'nvidia/llama-nemoretriever-colembed-3b-v1',
        device_map=f'cuda:{DEVICE}',
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        revision='50c36f4d5271c6851aa08bd26d69f6e7ca8b870c',
    ).eval()
    print("✓ 模型加载完成")
    
    # 2. 初始化Pipeline
    print("\n2. 初始化RAG Pipeline...")
    rag_pipeline = NvidiaRAGPipeline(
        retriever_model=retriever_model,
        use_faiss=True,
        faiss_config={"index_type": "flat"},
        device=DEVICE
    )
    print("✓ Pipeline初始化完成")
    
    # 3. 加载多个文档（使用较小的页面范围进行测试）
    print("\n3. 加载测试文档...")
    documents = {
        "tencent_esg": "contents/2024_Tencent_ESG[1-5].pdf",  # 只加载前5页
    }
    
    docid2images = {}
    for doc_id, pdf_path in documents.items():
        if os.path.exists(pdf_path):
            images = convert_from_path(pdf_path, dpi=200)
            docid2images[doc_id] = images[:5]  # 只取前5页
            print(f"   ✓ {doc_id}: {len(docid2images[doc_id])} 页")
        else:
            print(f"   ⚠️ 文件不存在: {pdf_path}")
    
    if not docid2images:
        print("⚠️ 没有找到测试文档，跳过Pipeline测试")
        return
    
    # 4. 编码文档
    print("\n4. 编码文档...")
    docid2embeddings = rag_pipeline.encode_documents(
        docid2images=docid2images,
        batch_size=4
    )
    print("✓ 文档编码完成")
    
    # 5. 构建索引
    print("\n5. 构建统一索引...")
    rag_pipeline.build_unified_index()
    print("✓ 索引构建完成")
    
    # 6. 测试检索
    print("\n6. 测试多文档检索...")
    queries = ["公司的可持续发展战略是什么？"]
    
    # 跨文档跨页面检索
    print("\n   模式1: 跨文档跨页面检索")
    results = rag_pipeline.retrieve_multi_doc(
        queries=queries,
        top_k=3,
        single_page_per_doc=False
    )
    
    for i, (query, query_results) in enumerate(zip(queries, results)):
        print(f"\n   查询: {query}")
        for result in query_results:
            doc_id = result.get('doc_id', 'unknown')
            page_num = result['page_num']
            score = result['score']
            print(f"      - {doc_id}, 第 {page_num} 页, 分数: {score:.4f}")
    
    print("\n✓ 多文档检索测试通过")

if __name__ == "__main__":
    print("\n" + "="*80)
    print("多文档检索功能测试")
    print("="*80)
    
    # 测试工具函数
    test_utility_functions()
    
    # 测试Pipeline（需要GPU和模型）
    try:
        test_multi_doc_pipeline()
    except Exception as e:
        print(f"\n⚠️ Pipeline测试失败: {e}")
        print("   (可能需要GPU或模型文件)")
    
    print("\n" + "="*80)
    print("测试完成")
    print("="*80)
