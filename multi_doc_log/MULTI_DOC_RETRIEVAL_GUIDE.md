# 📚 多文档检索功能指南

## 🎯 概述

基于 M3DocRAG 的设计理念，`nvidia_rag_with_faiss.py` 现已支持多文档检索功能，可以同时检索多个文档并支持两种检索策略。

---

## 🚀 核心功能

### 1. 多文档编码

```python
from nvidia_rag_with_faiss import NvidiaRAGPipeline

# 准备多个文档的图片
docid2images = {
    "tencent_esg": [image1, image2, ...],  # 腾讯ESG报告
    "tcl_report": [image1, image2, ...],   # TCL年报
    "alibaba_esg": [image1, image2, ...]   # 阿里巴巴ESG报告
}

# 编码所有文档
docid2embeddings = rag_pipeline.encode_documents(
    docid2images=docid2images,
    batch_size=8
)
```

### 2. 构建统一索引

```python
# 构建统一的多文档FAISS索引
rag_pipeline.build_unified_index(save_dir="./faiss_index/multi_doc")
```

### 3. 两种检索模式

#### 模式1：跨文档跨页面检索（默认）

允许从同一文档返回多页，适合答案分散在多页的场景。

```python
results = rag_pipeline.retrieve_multi_doc(
    queries=["2022年员工总数是多少？"],
    top_k=10,
    single_page_per_doc=False  # 允许同一文档多页
)

# 结果示例:
# [
#   [
#     {'doc_id': 'tencent_esg', 'page_num': 77, 'score': 10.43, 'rank': 1},
#     {'doc_id': 'tencent_esg', 'page_num': 71, 'score': 10.38, 'rank': 2},
#     {'doc_id': 'tcl_report', 'page_num': 45, 'score': 10.25, 'rank': 3},
#     ...
#   ]
# ]
```

**特点**：
- ✅ 可以从同一文档返回多页
- ✅ 适合答案分散在多页的场景
- ✅ 真正的 multi-page 检索

#### 模式2：每文档单页检索

每个文档只返回得分最高的一页，保证文档多样性。

```python
results = rag_pipeline.retrieve_multi_doc(
    queries=["比较不同公司的碳排放数据"],
    top_k=5,
    single_page_per_doc=True  # 每个文档只返回一页
)

# 结果示例:
# [
#   [
#     {'doc_id': 'tencent_esg', 'page_num': 22, 'score': 20.9, 'rank': 1},
#     {'doc_id': 'alibaba_esg', 'page_num': 18, 'score': 19.5, 'rank': 2},
#     {'doc_id': 'tcl_report', 'page_num': 30, 'score': 18.2, 'rank': 3},
#     ...
#   ]
# ]
```

**特点**：
- ✅ 保证文档多样性
- ✅ 适合答案在不同文档的场景
- ✅ 每个文档只返回最相关的一页

## 📊 应用场景

### 场景1：跨年度信息聚合

**问题**："2022年和2023年的员工总数分别是多少？"

**检索结果**：
```python
[
    {'doc_id': 'annual_report_2022', 'page_num': 77, 'score': 15.4},
    {'doc_id': 'annual_report_2023', 'page_num': 83, 'score': 14.8},
]
```

**优势**：可以同时检索到两年的信息

### 场景2：多公司对比

**问题**："比较腾讯和阿里巴巴的碳排放数据"

**检索结果**：
```python
[
    {'doc_id': 'tencent_esg', 'page_num': 22, 'score': 20.9},
    {'doc_id': 'alibaba_esg', 'page_num': 18, 'score': 19.5},
]
```

**优势**：可以从不同文档检索相关信息

### 场景3：长文档深度检索

**问题**："温室气体排放的详细分类"

**检索结果**（single_page_per_doc=False）：
```python
[
    {'doc_id': 'esg_report', 'page_num': 22, 'score': 20.9},  # 总览页
    {'doc_id': 'esg_report', 'page_num': 95, 'score': 20.3},  # 详细数据页
    {'doc_id': 'esg_report', 'page_num': 29, 'score': 20.1},  # 分类说明页
]
```

**优势**：可以从同一文档返回多个相关页面


## 📝 API 参考

### NvidiaRAGPipeline.encode_documents()

```python
def encode_documents(
    self,
    docid2images: Dict[str, List[Image.Image]],
    batch_size: int = 8
) -> Dict[str, torch.Tensor]
```

**参数**：
- `docid2images`: {doc_id: [images]} 文档ID到图片列表的映射
- `batch_size`: 批处理大小

**返回**：
- `docid2embeddings`: {doc_id: embeddings}

### NvidiaRAGPipeline.build_unified_index()
## 💡 最佳实践

1. **文档命名**：使用有意义的 `doc_id`，例如 `"tencent_esg_2024"` 而不是 `"doc1"`

2. **索引保存**：对于大规模文档集合，建议保存索引以便重复使用

3. **检索策略选择**：
   - 需要深度分析单个文档 → `single_page_per_doc=False`
   - 需要对比多个文档 → `single_page_per_doc=True`

4. **批处理大小**：根据GPU显存调整 `batch_size`，建议值：4-8


