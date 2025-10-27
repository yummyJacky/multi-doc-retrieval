# Nvidia_FAISS_MultiDoc.ipynb 更新总结

## 📝 更新内容

为多文档检索notebook添加了完整的Reranking和VQA功能，使其与单文档版本功能对齐。

---

## 🆕 新增功能

### 1. Reranker模型加载（Cell 21-22）

```python
from nvidia_rag_with_faiss import ImageReranker

# 初始化 Reranker
reranker = ImageReranker(
    model_name="monovlm",
    device=f"cuda:{DEVICE}",
    use_fast=True
)
```

**特点**：
- 使用MonoVLM模型
- 支持GPU加速
- 快速处理模式

---

### 2. 批量Reranking（Cell 23-25）

```python
# 准备候选图片（从多个文档中收集）
all_candidate_images = []
for query_results in results:
    candidate_images = []
    for result in query_results:
        doc_id = result['doc_id']
        page_idx = result['page_idx']
        # 从对应的文档中获取图片
        candidate_images.append(docid2images[doc_id][page_idx])
    all_candidate_images.append(candidate_images)

# 批量 reranking
all_rerank_results = reranker.rerank_batch(
    queries=queries,
    all_images_list=all_candidate_images,
    top_k=10
)
```

**关键改进**：
- ✅ 支持多文档场景
- ✅ 从不同文档中收集候选页面
- ✅ 使用`doc_id`和`page_idx`定位图片
- ✅ 批量处理提升效率

---

### 3. VQA问答（Cell 26-28）

```python
# 初始化 VQA
vqa_model = VQAModel(
    model_name="doubao-seed-1-6-vision-250815"
)

# 批量问答（每个查询使用 top-5 图片）
answers = vqa_model.answer_batch_with_multiple_images(
    queries=queries,
    all_images_list=all_top5_images,
    max_tokens=512
)
```

**特点**：
- ✅ 使用豆包多模态模型
- ✅ 基于重排序后的top-5页面
- ✅ 支持跨文档综合分析
- ✅ 批量处理多个查询

---

## 🔄 与单文档版本的对比

| 功能 | 单文档版本 | 多文档版本（更新后） |
|------|-----------|-------------------|
| **检索** | ✅ 单文档检索 | ✅ 多文档检索（跨页面/单页） |
| **Reranking** | ✅ 支持 | ✅ 支持（多文档） |
| **VQA** | ✅ 支持 | ✅ 支持（跨文档） |
| **图片收集** | 直接索引 | 通过`doc_id`+`page_idx` |
| **结果展示** | 页码 | 文档ID + 页码 |

---

## 📊 完整流程

### 1. 多文档检索

```python
results = rag_pipeline.retrieve_multi_doc(
    queries=queries,
    top_k=20,
    single_page_per_doc=False  # 跨文档跨页面
)
```

**输出格式**：
```python
{
    'query': '查询问题',
    'doc_id': 'sanqi_esg',      # 文档ID
    'page_idx': 81,              # 页面索引（0-based）
    'page_num': 82,              # 页面编号（1-based）
    'page_id': 'sanqi_esg_page81',
    'score': 12.6012,
    'rank': 1
}
```

### 2. 收集候选图片

```python
all_candidate_images = []
for query_results in results:
    candidate_images = []
    for result in query_results:
        doc_id = result['doc_id']
        page_idx = result['page_idx']
        # 从对应文档中获取图片
        candidate_images.append(docid2images[doc_id][page_idx])
    all_candidate_images.append(candidate_images)
```

**关键点**：
- 使用`doc_id`定位文档
- 使用`page_idx`定位页面
- 支持跨文档收集

### 3. Reranking

```python
all_rerank_results = reranker.rerank_batch(
    queries=queries,
    all_images_list=all_candidate_images,
    top_k=10
)
```

**输出**：
```
查询 1: 针对GRI403-3，各个公司有什么区别？
Reranking Top-10:
  1. 文档: sanqi_esg, 第 82 页
      Rerank分数: 0.9500, 检索分数: 12.6012
  2. 文档: archi_esg, 第 48 页
      Rerank分数: 0.8800, 检索分数: 9.6165
  ...
```

### 4. VQA问答

```python
answers = vqa_model.answer_batch_with_multiple_images(
    queries=queries,
    all_images_list=all_top5_images,
    max_tokens=512
)
```

**输出**：
```
问题 1: 针对GRI403-3，各个公司有什么区别？
使用页面: Top-5 综合分析
  1. 文档: sanqi_esg, 第 82 页
  2. 文档: archi_esg, 第 48 页
  3. 文档: zhongxing_esg, 第 142 页
  4. 文档: tencent_esg, 第 111 页
  5. 文档: sanqi_esg, 第 84 页

答案:
[多模态LLM生成的答案，综合分析多个文档的内容]
```

---

## 🎯 技术亮点

### 1. 多文档图片定位

**单文档**：
```python
images[page_idx]  # 直接索引
```

**多文档**：
```python
docid2images[doc_id][page_idx]  # 两级索引
```

### 2. 结果结构

**检索结果包含文档信息**：
- `doc_id`: 文档标识
- `page_idx`: 页面索引
- `page_num`: 页面编号
- `page_id`: 完整页面ID（`{doc_id}_page{page_idx}`）

### 3. 跨文档分析

VQA模型可以：
- ✅ 同时分析来自不同文档的页面
- ✅ 比较不同公司的数据
- ✅ 综合多个来源的信息

---

## 📋 使用示例

### 完整流程

```python
# 1. 定义查询
queries = ['针对GRI403-3，各个公司有什么区别？']

# 2. 多文档检索
results = rag_pipeline.retrieve_multi_doc(
    queries=queries,
    top_k=20,
    single_page_per_doc=False
)

# 3. 收集候选图片
all_candidate_images = []
for query_results in results:
    candidate_images = [
        docid2images[r['doc_id']][r['page_idx']] 
        for r in query_results
    ]
    all_candidate_images.append(candidate_images)

# 4. Reranking
all_rerank_results = reranker.rerank_batch(
    queries=queries,
    all_images_list=all_candidate_images,
    top_k=10
)

# 5. 提取top-5图片
all_top5_images = [
    [all_candidate_images[i][rr['doc_id']] for rr in rerank_results[:5]]
    for i, rerank_results in enumerate(all_rerank_results)
]

# 6. VQA问答
answers = vqa_model.answer_batch_with_multiple_images(
    queries=queries,
    all_images_list=all_top5_images,
    max_tokens=512
)
```

---

## 🔧 关键修改点

### 1. 图片收集逻辑

**之前（单文档）**：
```python
candidate_images = [images[r['page_index']] for r in results]
```

**现在（多文档）**：
```python
candidate_images = [
    docid2images[r['doc_id']][r['page_idx']] 
    for r in query_results
]
```

### 2. 结果展示

**之前（单文档）**：
```python
print(f"第 {page_num} 页")
```

**现在（多文档）**：
```python
print(f"文档: {doc_id}, 第 {page_num} 页")
```

---

## ✅ 验证清单

- [x] Reranker模型加载
- [x] 多文档图片收集
- [x] 批量Reranking
- [x] Top-5图片提取
- [x] VQA模型初始化
- [x] 批量问答
- [x] 结果展示（包含文档ID）
- [x] 更新总结部分

---

## 📚 相关文档

- **Nvidia_FAISS.ipynb** - 单文档版本参考
- **MULTI_DOC_CORRECTION_SUMMARY.md** - 多文档检索修正总结
- **RETRIEVE_MULTI_DOC_BUG_FIX.md** - 检索bug修复文档

---

## 🎓 学习要点

### 1. 多文档数据结构

```python
docid2images = {
    "tencent_esg": [img1, img2, ...],
    "sanqi_esg": [img1, img2, ...],
    ...
}
```

### 2. 检索结果结构

```python
result = {
    'doc_id': str,      # 文档ID
    'page_idx': int,    # 页面索引（0-based）
    'page_num': int,    # 页面编号（1-based）
    'score': float,     # 检索分数
    'rank': int         # 排名
}
```

### 3. 两级索引

```python
# 文档级别
docid2images[doc_id]

# 页面级别
docid2images[doc_id][page_idx]
```

---

**更新完成时间**: 2025-10-27  
**更新内容**: 添加Reranking和VQA功能  
**状态**: ✅ 完成
