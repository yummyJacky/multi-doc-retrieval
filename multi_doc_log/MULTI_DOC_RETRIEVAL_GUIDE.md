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

---

## 📖 完整使用示例

### Step 1: 初始化

```python
import torch
from transformers import AutoModel
from pdf2image import convert_from_path
from nvidia_rag_with_faiss import NvidiaRAGPipeline

# 加载模型
retriever_model = AutoModel.from_pretrained(
    'nvidia/llama-nemoretriever-colembed-3b-v1',
    device_map='cuda:9',
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
).eval()

# 初始化Pipeline
rag_pipeline = NvidiaRAGPipeline(
    retriever_model=retriever_model,
    use_faiss=True,
    faiss_config={"index_type": "ivfflat", "nlist": 10},
    device=9
)
```

### Step 2: 加载多个文档

```python
# 定义文档
documents = {
    "tencent_esg": "contents/2024_Tencent_ESG.pdf",
    "tcl_report": "contents/TCL2023.pdf",
}

# 加载图片
docid2images = {}
for doc_id, pdf_path in documents.items():
    images = convert_from_path(pdf_path, dpi=200)
    docid2images[doc_id] = images
    print(f"{doc_id}: {len(images)} 页")
```

### Step 3: 编码和索引

```python
# 编码所有文档
docid2embeddings = rag_pipeline.encode_documents(
    docid2images=docid2images,
    batch_size=8
)

# 构建统一索引
rag_pipeline.build_unified_index(save_dir="./faiss_index/multi_doc")
```

### Step 4: 检索

```python
queries = [
    '2022年员工总数是多少？',
    '截至二零二四年温室气体排放总量是多少？',
    '公司的可持续发展战略是什么？'
]

# 跨文档跨页面检索
results = rag_pipeline.retrieve_multi_doc(
    queries=queries,
    top_k=10,
    single_page_per_doc=False
)

# 显示结果
for i, (query, query_results) in enumerate(zip(queries, results)):
    print(f"\n查询 {i+1}: {query}")
    for result in query_results[:5]:
        doc_id = result['doc_id']
        page_num = result['page_num']
        score = result['score']
        print(f"  - {doc_id}, 第 {page_num} 页, 分数: {score:.4f}")
```

---

## 🔧 工具函数

### get_top_k_pages()

从所有文档的所有页面中选择 top-k（跨文档跨页面）。

```python
from nvidia_rag_with_faiss import get_top_k_pages

docid2scores = {
    "doc1": [10, 50, 30],
    "doc2": [40, 20, 60],
    "doc3": [70, 90]
}

top_pages = get_top_k_pages(docid2scores, k=3)
# 返回: [('doc3', 1, 90), ('doc3', 0, 70), ('doc2', 2, 60)]
```

### get_top_k_pages_single_page_from_each_doc()

每个文档只返回得分最高的一页。

```python
from nvidia_rag_with_faiss import get_top_k_pages_single_page_from_each_doc

top_pages = get_top_k_pages_single_page_from_each_doc(docid2scores, k=2)
# 返回: [('doc3', 1, 90), ('doc2', 2, 60)]
```

---

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

---

## 🆚 与 M3DocRAG 的对比

| 特性 | M3DocRAG | 本实现 |
|------|----------|--------|
| **多文档支持** | ✅ 原生支持 | ✅ 完全支持 |
| **页面标识** | `doc_id_pageN` | `doc_id_page_N` |
| **检索模式** | 两种模式 | 两种模式 |
| **嵌入维度** | 128 (ColPali) | 3072 (Nvidia NemoRetriever) |
| **索引结构** | FAISS + token2pageuid | FAISS + token2page 映射 |
| **工具函数** | ✅ | ✅ 完全兼容 |

---

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

```python
def build_unified_index(
    self,
    save_dir: Optional[str] = None
)
```

**参数**：
- `save_dir`: 索引保存目录（可选）

### NvidiaRAGPipeline.retrieve_multi_doc()

```python
def retrieve_multi_doc(
    self,
    queries: List[str],
    top_k: int = 10,
    single_page_per_doc: bool = False,
    use_index: bool = True,
    batch_size: int = 8
) -> List[List[Dict]]
```

**参数**：
- `queries`: 查询文本列表
- `top_k`: 返回top-k个页面
- `single_page_per_doc`: 是否每个文档只返回一页
- `use_index`: 是否使用FAISS索引
- `batch_size`: 批处理大小

**返回**：
- `List[List[Dict]]`: 每个查询对应一个结果列表

---

## 💡 最佳实践

1. **文档命名**：使用有意义的 `doc_id`，例如 `"tencent_esg_2024"` 而不是 `"doc1"`

2. **索引保存**：对于大规模文档集合，建议保存索引以便重复使用

3. **检索策略选择**：
   - 需要深度分析单个文档 → `single_page_per_doc=False`
   - 需要对比多个文档 → `single_page_per_doc=True`

4. **批处理大小**：根据GPU显存调整 `batch_size`，建议值：4-8

---

## 🎓 示例Notebook

完整的可运行示例请参考：
- `Nvidia_FAISS_MultiDoc.ipynb` - 多文档检索完整示例

---

## 📚 参考资料

- **M3DocRAG 分析文档**: `M3DOCRAG_MULTIPAGE_ANALYSIS.md`
- **M3DocRAG 论文**: [Multi-modal Retrieval is What You Need for Multi-page Multi-document Understanding](https://m3docrag.github.io/)
- **Nvidia NemoRetriever**: [nvidia/llama-nemoretriever-colembed-3b-v1](https://huggingface.co/nvidia/llama-nemoretriever-colembed-3b-v1)

---

## 🔄 更新日志

### v1.0 (2024-10-24)
- ✅ 添加多文档编码支持
- ✅ 添加统一索引构建
- ✅ 实现两种检索模式
- ✅ 添加工具函数 `get_top_k_pages` 和 `get_top_k_pages_single_page_from_each_doc`
- ✅ 完全兼容 M3DocRAG 的设计理念
