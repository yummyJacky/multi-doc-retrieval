# 更新日志 - 多文档检索功能

## [1.0.0] - 2024-10-24

### 🎉 新增功能

#### 核心功能
- **多文档编码**: 添加 `encode_documents()` 方法，支持同时编码多个文档
- **统一索引**: 添加 `build_unified_index()` 方法，构建跨文档的统一FAISS索引
- **多文档检索**: 添加 `retrieve_multi_doc()` 方法，支持两种检索模式
  - 跨文档跨页面检索（允许同一文档返回多页）
  - 每文档单页检索（保证文档多样性）

#### 工具函数
- **`get_top_k_pages()`**: 从所有文档的所有页面中选择 top-k
- **`get_top_k_pages_single_page_from_each_doc()`**: 每个文档只返回得分最高的一页

#### 数据结构
- 添加 `docid2embeddings: Dict[str, torch.Tensor]` 存储多文档嵌入
- 添加 `docid2page_ids: Dict[str, List[str]]` 存储多文档页面ID
- 添加 `is_multi_doc_mode: bool` 标志多文档模式

### 页面ID命名规范

- **单文档模式**: `page_0`, `page_1`, `page_2`, ...
- **多文档模式**: `doc_id_page_0`, `doc_id_page_1`, ...

例如：
- `tencent_esg_page_76` - 腾讯ESG报告第77页（0-indexed）
- `tcl_report_page_0` - TCL年报第1页

## 使用示例

### 快速开始

```python
from nvidia_rag_with_faiss import NvidiaRAGPipeline

# 1. 初始化
rag_pipeline = NvidiaRAGPipeline(
    retriever_model=retriever_model,
    use_faiss=True,
    device=9
)

# 2. 编码多个文档
docid2images = {
    "doc1": [image1, image2, ...],
    "doc2": [image1, image2, ...],
}
rag_pipeline.encode_documents(docid2images)

# 3. 构建统一索引
rag_pipeline.build_unified_index(save_dir="./faiss_index/multi_doc")

# 4. 多文档检索
results = rag_pipeline.retrieve_multi_doc(
    queries=["问题"],
    top_k=10,
    single_page_per_doc=False
)
```

### 两种检索模式对比

```python
# 模式1: 跨文档跨页面（允许同一文档多页）
results = rag_pipeline.retrieve_multi_doc(
    queries=["温室气体排放的详细分类"],
    top_k=10,
    single_page_per_doc=False
)
# 可能返回: doc1_page22, doc1_page95, doc1_page29, doc2_page15, ...

# 模式2: 每文档单页（保证文档多样性）
results = rag_pipeline.retrieve_multi_doc(
    queries=["比较不同公司的碳排放数据"],
    top_k=5,
    single_page_per_doc=True
)
# 可能返回: doc1_page22, doc2_page18, doc3_page30, ...
```

---

## 下一步计划

### 短期 (v1.1)
- [ ] 添加文档权重支持
- [ ] 优化大规模文档的内存使用
- [ ] 添加更多的检索策略

### 中期 (v1.2)
- [ ] 支持增量索引更新
- [ ] 添加文档过滤功能
- [ ] 支持时间范围过滤
- [ ] 添加更多 Reranker 模型支持
- [ ] 实现本地 VQA 模型加载
- [ ] 添加结果缓存机制

### 长期 (v2.0)
- [ ] 支持分布式检索
- [ ] 添加查询扩展功能
- [ ] 集成更多的检索模型
- [ ] 增量索引更新
- [ ] 混合检索策略

---

