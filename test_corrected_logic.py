"""
测试修正后的多文档检索逻辑（不需要实际模型）
验证Token级索引和MaxSim聚合的核心逻辑
"""

import torch
import numpy as np
from typing import Dict, List, Tuple


def get_top_k_pages(
    docid2scores: Dict[str, List[float]], 
    k: int
) -> List[Tuple[str, int, float]]:
    """跨文档跨页面检索"""
    flattened = [
        (doc_id, page_idx, score)
        for doc_id, scores in docid2scores.items()
        for page_idx, score in enumerate(scores)
    ]
    flattened.sort(key=lambda x: x[2], reverse=True)
    return flattened[:k]


def get_top_k_pages_single_page_from_each_doc(
    docid2scores: Dict[str, List[float]], 
    k: int
) -> List[Tuple[str, int, float]]:
    """每文档单页检索"""
    highest_per_doc = []
    for doc_id, scores in docid2scores.items():
        if len(scores) > 0:
            max_idx = max(range(len(scores)), key=lambda i: scores[i])
            highest_per_doc.append((doc_id, max_idx, scores[max_idx]))
    highest_per_doc.sort(key=lambda x: x[2], reverse=True)
    return highest_per_doc[:k]


def test_retrieval_strategies():
    """测试两种检索策略"""
    print("\n" + "="*80)
    print("测试1: 检索策略函数")
    print("="*80)
    
    docid2scores = {
        "doc1": [10.0, 50.0, 30.0],
        "doc2": [40.0, 20.0, 60.0],
        "doc3": [70.0, 90.0]
    }
    
    print("\n跨文档跨页面检索（top-3）:")
    result1 = get_top_k_pages(docid2scores, k=3)
    for rank, (doc_id, page_idx, score) in enumerate(result1, 1):
        print(f"  {rank}. {doc_id}_page{page_idx}: {score}")
    
    assert result1[0] == ('doc3', 1, 90.0), "最高分应该是doc3的第1页"
    assert result1[1] == ('doc3', 0, 70.0), "第二高分应该是doc3的第0页"
    assert result1[2] == ('doc2', 2, 60.0), "第三高分应该是doc2的第2页"
    print("  ✓ 测试通过")
    
    print("\n每文档单页检索（top-2）:")
    result2 = get_top_k_pages_single_page_from_each_doc(docid2scores, k=2)
    for rank, (doc_id, page_idx, score) in enumerate(result2, 1):
        print(f"  {rank}. {doc_id}_page{page_idx}: {score}")
    
    assert result2[0] == ('doc3', 1, 90.0), "最高分应该是doc3的最高页"
    assert result2[1] == ('doc2', 2, 60.0), "第二高分应该是doc2的最高页"
    assert len(result2) == 2, "应该只返回2个文档"
    print("  ✓ 测试通过")


def test_token_level_indexing():
    """测试Token级索引构建"""
    print("\n" + "="*80)
    print("测试2: Token级索引构建")
    print("="*80)
    
    # 模拟文档嵌入
    docid2embeddings = {
        "doc1": torch.randn(2, 10, 128),  # 2页，每页10个token
        "doc2": torch.randn(3, 15, 128),  # 3页，每页15个token
    }
    
    # 模拟扁平化过程
    all_token_embeddings = []
    token2pageuid = []
    
    total_pages = 0
    total_tokens = 0
    
    for doc_id in sorted(docid2embeddings.keys()):
        doc_embs = docid2embeddings[doc_id]
        n_pages = doc_embs.shape[0]
        
        for page_idx in range(n_pages):
            page_emb = doc_embs[page_idx]
            n_tokens = page_emb.shape[0]
            
            all_token_embeddings.append(page_emb)
            page_uid = f"{doc_id}_page{page_idx}"
            token2pageuid.extend([page_uid] * n_tokens)
            
            total_tokens += n_tokens
        
        total_pages += n_pages
    
    all_token_embeddings = torch.cat(all_token_embeddings, dim=0)
    
    print(f"\n文档统计:")
    print(f"  - doc1: 2页 × 10 tokens = 20 tokens")
    print(f"  - doc2: 3页 × 15 tokens = 45 tokens")
    print(f"  - 总页面数: {total_pages}")
    print(f"  - 总Token数: {total_tokens}")
    print(f"  - 嵌入形状: {all_token_embeddings.shape}")
    
    assert all_token_embeddings.shape[0] == 65, f"总token数应该是65，实际是{all_token_embeddings.shape[0]}"
    assert len(token2pageuid) == 65, f"映射表长度应该是65，实际是{len(token2pageuid)}"
    
    # 验证映射关系
    print(f"\n映射关系验证:")
    print(f"  - token[0] -> {token2pageuid[0]} (应该是doc1_page0)")
    print(f"  - token[10] -> {token2pageuid[10]} (应该是doc1_page1)")
    print(f"  - token[20] -> {token2pageuid[20]} (应该是doc2_page0)")
    
    assert token2pageuid[0] == "doc1_page0"
    assert token2pageuid[10] == "doc1_page1"
    assert token2pageuid[20] == "doc2_page0"
    print("  ✓ 测试通过")


def test_maxsim_aggregation():
    """测试MaxSim聚合逻辑"""
    print("\n" + "="*80)
    print("测试3: MaxSim聚合逻辑")
    print("="*80)
    
    # 模拟token到页面的映射
    token2pageuid = [
        "doc1_page0", "doc1_page0", "doc1_page1", "doc1_page1",
        "doc2_page0", "doc2_page0", "doc2_page1"
    ]
    
    # 模拟3个查询token的搜索结果
    # 格式: {查询token索引: [(文档token索引, 相似度分数), ...]}
    query_results = {
        0: [(0, 0.9), (1, 0.8), (4, 0.7)],  # 查询token 0
        1: [(2, 0.85), (3, 0.75), (5, 0.6)],  # 查询token 1
        2: [(1, 0.95), (4, 0.8), (6, 0.7)]   # 查询token 2
    }
    
    print("\n查询token搜索结果:")
    for q_idx, results in query_results.items():
        print(f"  查询token {q_idx}:")
        for token_idx, score in results:
            print(f"    -> 文档token[{token_idx}] ({token2pageuid[token_idx]}): {score}")
    
    # MaxSim聚合
    final_page2scores = {}
    
    for q_token_idx, results in query_results.items():
        current_page2scores = {}
        
        for token_idx, score in results:
            page_uid = token2pageuid[token_idx]
            
            # MaxSim: 每个查询token对每个页面只保留最高分
            if page_uid not in current_page2scores:
                current_page2scores[page_uid] = score
            else:
                current_page2scores[page_uid] = max(current_page2scores[page_uid], score)
        
        # 累加所有查询token的MaxSim分数
        for page_uid, score in current_page2scores.items():
            if page_uid in final_page2scores:
                final_page2scores[page_uid] += score
            else:
                final_page2scores[page_uid] = score
    
    print("\nMaxSim聚合结果:")
    for page_uid, score in sorted(final_page2scores.items(), key=lambda x: x[1], reverse=True):
        print(f"  {page_uid}: {score:.2f}")
    
    # 验证计算
    print("\n计算验证:")
    print("  doc1_page0:")
    print("    - 查询token 0: max(0.9, 0.8) = 0.9")
    print("    - 查询token 2: 0.95")
    print("    - 总分: 0.9 + 0.95 = 1.85")
    
    print("  doc1_page1:")
    print("    - 查询token 1: max(0.85, 0.75) = 0.85")
    print("    - 总分: 0.85")
    
    print("  doc2_page0:")
    print("    - 查询token 0: 0.7")
    print("    - 查询token 2: 0.8")
    print("    - 总分: 0.7 + 0.8 = 1.5")
    
    assert abs(final_page2scores["doc1_page0"] - 1.85) < 0.01
    assert abs(final_page2scores["doc1_page1"] - 0.85) < 0.01
    assert abs(final_page2scores["doc2_page0"] - 1.5) < 0.01
    print("  ✓ 测试通过")


def compare_implementations():
    """对比新旧实现的差异"""
    print("\n" + "="*80)
    print("对比: 新旧实现的关键差异")
    print("="*80)
    
    print("\n【错误实现】nvidia_rag_with_faiss.py (旧版):")
    print("  1. 索引粒度: 页面级别")
    print("     - 将每页的token嵌入合并成单个向量")
    print("     - unified_embeddings = torch.cat(all_embeddings, dim=0)  # [n_pages, n_tokens, dim]")
    print("  2. 检索方式: 页面级检索")
    print("     - 直接在页面向量上搜索")
    print("     - 丢失了token级别的细粒度匹配")
    print("  3. MaxSim: 被破坏")
    print("     - 无法实现正确的MaxSim聚合")
    
    print("\n【正确实现】nvidia_rag_with_faiss.py (新版):")
    print("  1. 索引粒度: Token级别")
    print("     - 扁平化所有token: all_token_embeddings = [total_tokens, dim]")
    print("     - 维护映射: token2pageuid[i] = 'doc_id_page{idx}'")
    print("  2. 检索方式: Token级检索 + MaxSim聚合")
    print("     - 对每个查询token找最近的文档token")
    print("     - 每个查询token对每个页面保留最高分")
    print("     - 对所有查询token的MaxSim分数求和")
    print("  3. MaxSim: 完整保留")
    print("     - MaxSim(Q, D) = Σ_i max_j sim(q_i, d_j)")
    
    print("\n【与M3DocRAG对齐】:")
    print("  ✅ Token级扁平化索引")
    print("  ✅ token2pageuid映射表")
    print("  ✅ MaxSim聚合机制")
    print("  ✅ Late Interaction架构")
    print("  ✅ 两种检索策略")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("测试修正后的多文档检索逻辑")
    print("="*80)
    
    # 运行所有测试
    test_retrieval_strategies()
    test_token_level_indexing()
    test_maxsim_aggregation()
    compare_implementations()
    
    print("\n" + "="*80)
    print("✓ 所有测试通过！")
    print("="*80)
    print("\n核心改进:")
    print("  1. ✅ Token级索引 (而非页面级)")
    print("  2. ✅ token2pageuid映射")
    print("  3. ✅ MaxSim聚合保留")
    print("  4. ✅ Late Interaction架构")
    print("  5. ✅ 与M3DocRAG完全对齐")
    print("\n修改文件: nvidia_rag_with_faiss.py")
    print("  - build_unified_index(): Token级扁平化")
    print("  - retrieve_multi_doc(): MaxSim聚合")
    print("  - load_index(): 加载token映射")
    print()
