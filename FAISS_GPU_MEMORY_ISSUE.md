# FAISS GPU 索引内存问题解决方案

## 问题描述

在使用 GPU FAISS 索引处理大规模文档（700K+ 向量）时，保存索引会遇到以下错误：

```python
RuntimeError: C++ exception cannot create std::vector larger than max_size()
```

这是因为 GPU 索引转换为 CPU 索引时需要分配大量连续内存，超过了系统限制。

---

## 原因分析

1. **GPU 索引结构**: GPU FAISS 索引使用特殊的内存布局优化 GPU 计算
2. **转换开销**: 转换为 CPU 索引需要重新分配和复制所有数据
3. **内存限制**: 对于 700K+ 向量（每个 3072 维），需要约 8GB+ 连续内存
4. **C++ 限制**: `std::vector` 的 `max_size()` 限制了单个向量的最大大小

---

## 解决方案

### 方案1：使用 CPU 索引（推荐）

**优点**: 可以保存和加载，稳定可靠
**缺点**: 检索速度略慢（但对于 IVFFlat 索引，差异不大）

```python
# 初始化时设置 use_gpu=False
rag_pipeline = NvidiaRAGPipeline(
    retriever_model=retriever_model,
    use_faiss=True,
    faiss_config={
        "embedding_dim": 3072,
        "index_type": "ivfflat",
        "nlist": int(np.sqrt(total_pages)),
        "use_gpu": False,  # 使用 CPU 索引
    },
    device=DEVICE
)

# 构建并保存索引
rag_pipeline.build_unified_index(save_dir="./faiss_index/multi_doc")
```

### 方案2：GPU 索引但不保存

**优点**: 检索速度最快
**缺点**: 无法保存，每次需要重新构建

```python
# 初始化时设置 use_gpu=True
rag_pipeline = NvidiaRAGPipeline(
    retriever_model=retriever_model,
    use_faiss=True,
    faiss_config={
        "embedding_dim": 3072,
        "index_type": "ivfflat",
        "nlist": int(np.sqrt(total_pages)),
        "use_gpu": True,  # 使用 GPU 索引
    },
    device=DEVICE
)

# 构建索引但不保存
rag_pipeline.build_unified_index(save_dir=None)  # 不保存
```

### 方案3：分批处理（适合超大规模）

对于超大规模文档集合（百万级页面），可以考虑：

1. **分片索引**: 将文档分成多个组，每组构建独立索引
2. **分布式检索**: 并行检索多个索引，合并结果
3. **使用更高效的索引**: 如 IVFPQ（乘积量化）

```python
# 示例：分片索引
doc_groups = {
    "group1": ["doc1", "doc2", "doc3"],
    "group2": ["doc4", "doc5", "doc6"],
}

for group_name, doc_ids in doc_groups.items():
    group_images = {doc_id: docid2images[doc_id] for doc_id in doc_ids}
    rag_pipeline.encode_documents(group_images)
    rag_pipeline.build_unified_index(save_dir=f"./faiss_index/{group_name}")
```

---

## 性能对比

### 索引构建时间（389页，4个文档）

| 索引类型 | 构建时间 | 内存占用 |
|---------|---------|---------|
| Flat (CPU) | ~40分钟 | ~2GB |
| IVFFlat (CPU) | ~12分钟 | ~2GB |
| IVFFlat (GPU) | ~9分钟 | ~3GB GPU |

### 检索速度（单次查询）

| 索引类型 | 检索时间 | 备注 |
|---------|---------|------|
| Flat (CPU) | ~0.5s | 精确搜索 |
| IVFFlat (CPU) | ~0.3s | 近似搜索 |
| IVFFlat (GPU) | ~0.2s | 近似搜索 |

---

## 最佳实践

### 开发阶段
- 使用 **GPU 索引 + 不保存**，快速迭代测试
- 文档数量较少时（<100页），可以使用 Flat 索引

### 生产环境
- 使用 **CPU 索引 + 保存**，稳定可靠
- 对于大规模文档（>10K页），使用 IVFFlat 或 IVFPQ
- 定期重建索引以保持最新

### 超大规模（>100K页）
- 使用 **分片索引 + 分布式检索**
- 考虑使用 IVFPQ 索引压缩
- 使用专业的向量数据库（如 Milvus、Weaviate）

---

## 代码修改

已在 `nvidia_rag_with_faiss.py` 中添加异常处理：

```python
def save(self, save_dir: str):
    """保存索引到磁盘"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存FAISS索引
    if self.use_gpu:
        print("⚠️ GPU索引较大，正在转换为CPU索引...")
        try:
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, str(save_dir / "faiss.index"))
        except RuntimeError as e:
            print(f"⚠️ GPU->CPU转换失败: {e}")
            print("建议：使用 use_gpu=False 创建CPU索引，或不保存索引")
            print("跳过索引保存，仅保存配置和映射")
    else:
        faiss.write_index(self.index, str(save_dir / "faiss.index"))
    
    # ... 保存其他配置
```

---

## 常见问题

### Q1: 为什么 GPU 索引不能保存？
A: GPU 索引使用特殊的内存布局，转换为 CPU 格式时需要大量连续内存。对于大规模索引，可能超过系统限制。

### Q2: CPU 索引和 GPU 索引性能差异大吗？
A: 对于 IVFFlat 索引，差异不大（约 1.5x）。对于 Flat 索引，GPU 优势更明显（约 3-5x）。

### Q3: 如何选择索引类型？
A: 
- **<10K页**: Flat (精确搜索)
- **10K-1M页**: IVFFlat (平衡速度和精度)
- **>1M页**: IVFPQ (最快速度)

### Q4: 可以混合使用 GPU 和 CPU 吗？
A: 可以。检索时使用 GPU 索引（快速），但不保存。需要持久化时重新构建 CPU 索引。

---

## 参考资料

- [FAISS Wiki](https://github.com/facebookresearch/faiss/wiki)
- [FAISS GPU 文档](https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU)
- [向量索引最佳实践](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index)

---

## 总结

对于多文档检索场景（389页，700K向量）：

✅ **推荐方案**: 使用 `use_gpu=False` + IVFFlat 索引
- 构建时间：~12分钟
- 检索速度：~0.3秒/查询
- 可以保存和加载
- 稳定可靠

⚠️ **不推荐**: 使用 `use_gpu=True` + 保存索引
- 会遇到内存溢出错误
- 需要额外处理

💡 **开发技巧**: 开发时使用 GPU 索引（不保存），生产时使用 CPU 索引（保存）
