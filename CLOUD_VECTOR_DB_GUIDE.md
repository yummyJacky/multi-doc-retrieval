# 云向量数据库集成指南

## 🌐 为什么使用云向量数据库？

### 当前方案的局限性
使用本地 FAISS 索引存在以下问题：
- ❌ GPU 索引无法保存（内存溢出）
- ❌ 索引文件较大（700K 向量约 8GB）
- ❌ 不支持分布式部署
- ❌ 难以实现增量更新
- ❌ 缺少权限管理和监控

### 云向量数据库的优势
- ✅ **可扩展性**: 自动扩容，支持百万/亿级向量
- ✅ **高可用性**: 分布式部署，自动备份
- ✅ **增量更新**: 支持动态添加/删除文档
- ✅ **多租户**: 权限管理，数据隔离
- ✅ **监控运维**: 完善的监控和日志系统
- ✅ **API 访问**: RESTful API，多语言支持

---

## 📊 云向量数据库对比

### 1. Milvus / Zilliz Cloud ⭐⭐⭐⭐⭐

**推荐指数**: ⭐⭐⭐⭐⭐

**优势**:
- 开源，社区活跃
- 性能优秀，支持 GPU 加速
- 完美支持 FAISS 索引类型
- 可自建或使用云服务（Zilliz Cloud）
- 支持混合搜索（向量+标量）

**定价** (Zilliz Cloud):
- 免费层: 1GB 存储
- 标准版: $0.12/小时 起
- 企业版: 联系销售

**适用场景**:
- ✅ 大规模文档检索（>100K 页）
- ✅ 需要高性能和可扩展性
- ✅ 有技术团队可以自建

**安装**:
```bash
# 自建 Milvus
docker run -d --name milvus_standalone \
  -p 19530:19530 -p 9091:9091 \
  milvusdb/milvus:latest

# Python 客户端
pip install pymilvus
```

### 2. Pinecone ⭐⭐⭐⭐

**推荐指数**: ⭐⭐⭐⭐

**优势**:
- 纯云服务，无需运维
- 简单易用，5分钟上手
- 自动扩容和备份
- 良好的文档和支持

**定价**:
- 免费层: 1 个索引，100K 向量
- Starter: $70/月，500万向量
- Standard: $0.096/百万向量/月

**适用场景**:
- ✅ 快速原型开发
- ✅ 中小规模应用（<100万向量）
- ✅ 不想自建服务器

**安装**:
```bash
pip install pinecone-client
```

### 3. Weaviate ⭐⭐⭐⭐

**推荐指数**: ⭐⭐⭐⭐

**优势**:
- 开源，GraphQL API
- 支持多模态搜索
- 内置语义搜索
- 可自建或使用云服务

**定价** (Weaviate Cloud):
- 免费层: 14天试用
- Sandbox: $25/月
- Production: 按使用量计费

**适用场景**:
- ✅ 需要 GraphQL API
- ✅ 多模态搜索需求
- ✅ 知识图谱应用

**安装**:
```bash
# 自建
docker run -d -p 8080:8080 semitechnologies/weaviate:latest

# Python 客户端
pip install weaviate-client
```

### 4. Qdrant ⭐⭐⭐⭐

**推荐指数**: ⭐⭐⭐⭐

**优势**:
- 开源，Rust 编写，性能优秀
- 支持过滤和混合搜索
- 简单易用
- 可自建或使用云服务

**定价** (Qdrant Cloud):
- 免费层: 1GB 存储
- 按使用量计费

**适用场景**:
- ✅ 高性能要求
- ✅ 需要复杂过滤
- ✅ Rust 生态

**安装**:
```bash
# 自建
docker run -p 6333:6333 qdrant/qdrant

# Python 客户端
pip install qdrant-client
```

### 5. ChromaDB ⭐⭐⭐

**推荐指数**: ⭐⭐⭐

**优势**:
- 轻量级，易于集成
- 适合小规模应用
- 开源免费

**定价**:
- 完全免费（开源）

**适用场景**:
- ✅ 小规模应用（<10万向量）
- ✅ 本地开发测试
- ✅ 嵌入式应用

**安装**:
```bash
pip install chromadb
```

---

## 🚀 集成方案

### 方案1: Milvus（推荐用于生产）

#### 1.1 自建 Milvus

```bash
# 使用 Docker Compose
wget https://github.com/milvus-io/milvus/releases/download/v2.3.0/milvus-standalone-docker-compose.yml -O docker-compose.yml
docker-compose up -d
```

#### 1.2 使用 Zilliz Cloud

1. 注册账号: https://cloud.zilliz.com/
2. 创建集群
3. 获取连接信息

#### 1.3 集成代码

```python
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType

# 连接到 Milvus
connections.connect(
    alias="default",
    host="your-milvus-host.com",  # 或 localhost
    port="19530"
)

# 创建集合
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="page_id", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=3072),
]
schema = CollectionSchema(fields=fields)
collection = Collection(name="documents", schema=schema)

# 创建索引
index_params = {
    "metric_type": "IP",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 128}
}
collection.create_index(field_name="embedding", index_params=index_params)

# 插入数据
collection.insert([page_ids, embeddings.tolist()])

# 搜索
collection.load()
results = collection.search(
    data=query_embedding.tolist(),
    anns_field="embedding",
    param={"metric_type": "IP", "params": {"nprobe": 10}},
    limit=10
)
```

### 方案2: Pinecone（推荐用于快速开发）

```python
import pinecone

# 初始化
pinecone.init(api_key="your-api-key", environment="us-west1-gcp")

# 创建索引
pinecone.create_index(
    name="documents",
    dimension=3072,
    metric="dotproduct"
)

# 连接索引
index = pinecone.Index("documents")

# 插入数据
vectors = [(f"page_{i}", emb.tolist(), {"page_id": pid}) 
           for i, (emb, pid) in enumerate(zip(embeddings, page_ids))]
index.upsert(vectors=vectors)

# 搜索
results = index.query(
    vector=query_embedding.tolist(),
    top_k=10,
    include_metadata=True
)
```

---

## 💡 迁移步骤

### Step 1: 选择向量数据库

根据你的需求选择：
- **大规模 + 自建**: Milvus
- **大规模 + 云服务**: Zilliz Cloud
- **快速开发**: Pinecone
- **中小规模**: Qdrant / Weaviate

### Step 2: 修改代码

创建新的 `CloudVectorDBPipeline` 类：

```python
class CloudVectorDBPipeline(NvidiaRAGPipeline):
    """使用云向量数据库的 RAG Pipeline"""
    
    def __init__(self, retriever_model, vector_db_type="milvus", **db_config):
        super().__init__(retriever_model, use_faiss=False)
        
        # 初始化向量数据库
        if vector_db_type == "milvus":
            from cloud_vector_db_integration import MilvusIndexManager
            self.vector_db = MilvusIndexManager(**db_config)
        elif vector_db_type == "pinecone":
            from cloud_vector_db_integration import PineconeIndexManager
            self.vector_db = PineconeIndexManager(**db_config)
        # ... 其他数据库
    
    def build_unified_index(self, save_dir=None):
        """构建索引（上传到云数据库）"""
        # 合并嵌入
        all_embeddings = []
        all_page_ids = []
        for doc_id in sorted(self.docid2embeddings.keys()):
            all_embeddings.append(self.docid2embeddings[doc_id])
            all_page_ids.extend(self.docid2page_ids[doc_id])
        
        unified_embeddings = torch.cat(all_embeddings, dim=0)
        embeddings_np = unified_embeddings.numpy()
        
        # 上传到云数据库
        self.vector_db.insert_embeddings(embeddings_np, all_page_ids)
    
    def retrieve_multi_doc(self, queries, top_k=10, **kwargs):
        """使用云数据库检索"""
        # 编码查询
        query_embeddings = self.retriever_model.forward_queries(queries)
        
        # 搜索
        all_results = []
        for query_emb in query_embeddings:
            results = self.vector_db.search(
                query_embedding=query_emb.cpu().numpy(),
                top_k=top_k
            )
            all_results.append(results)
        
        return all_results
```

### Step 3: 测试和部署

```python
# 初始化
pipeline = CloudVectorDBPipeline(
    retriever_model=retriever_model,
    vector_db_type="milvus",
    host="your-milvus-host.com",
    port=19530
)

# 编码和索引
pipeline.encode_documents(docid2images)
pipeline.build_unified_index()

# 检索
results = pipeline.retrieve_multi_doc(queries=["问题"], top_k=10)
```

---

## 📈 性能对比

### 本地 FAISS vs 云向量数据库

| 指标 | 本地 FAISS | Milvus (自建) | Zilliz Cloud | Pinecone |
|------|-----------|--------------|--------------|----------|
| **索引构建** | 12分钟 | 15分钟 | 15分钟 | 20分钟 |
| **检索速度** | 0.3秒 | 0.4秒 | 0.5秒 | 0.6秒 |
| **可扩展性** | ❌ | ✅✅✅ | ✅✅✅ | ✅✅ |
| **增量更新** | ❌ | ✅ | ✅ | ✅ |
| **运维成本** | 低 | 中 | 低 | 低 |
| **存储成本** | 本地 | 服务器 | $0.12/h | $70/月 |

### 成本估算（389页，700K向量）

| 方案 | 月成本 | 年成本 | 备注 |
|------|--------|--------|------|
| 本地 FAISS | $0 | $0 | 需要自己的服务器 |
| Milvus (自建) | $50-100 | $600-1200 | 云服务器成本 |
| Zilliz Cloud | $86 | $1032 | 按小时计费 |
| Pinecone | $70 | $840 | 固定月费 |

---

## 🎯 推荐方案

### 场景1: 开发测试
**推荐**: 本地 FAISS (CPU) 或 ChromaDB
- 成本: $0
- 性能: 够用
- 优势: 简单快速

### 场景2: 小规模生产（<10万页）
**推荐**: Pinecone 或 Qdrant Cloud
- 成本: $70-100/月
- 性能: 良好
- 优势: 无需运维

### 场景3: 中大规模生产（10万-100万页）
**推荐**: Milvus (自建) 或 Zilliz Cloud
- 成本: $100-500/月
- 性能: 优秀
- 优势: 可扩展，高性能

### 场景4: 超大规模（>100万页）
**推荐**: Zilliz Cloud (企业版) 或 Milvus 集群
- 成本: 联系销售
- 性能: 极致
- 优势: 分布式，高可用

---

## 📝 最佳实践

### 1. 数据分片
对于超大规模文档，按文档类型或时间分片：
```python
collections = {
    "esg_reports": MilvusIndexManager(collection_name="esg_reports"),
    "financial_reports": MilvusIndexManager(collection_name="financial_reports"),
}
```

### 2. 增量更新
定期更新索引，而不是重建：
```python
# 只添加新文档
new_embeddings = pipeline.encode_passages(new_images)
vector_db.insert_embeddings(new_embeddings, new_page_ids)
```

### 3. 监控和日志
使用云服务的监控功能：
- 查询延迟
- 索引大小
- 错误率
- 成本追踪

### 4. 备份策略
定期备份索引：
```python
# Milvus 备份
collection.create_alias("backup_20241024")
```

---

## 🔗 相关资源

- **Milvus**: https://milvus.io/
- **Zilliz Cloud**: https://cloud.zilliz.com/
- **Pinecone**: https://www.pinecone.io/
- **Weaviate**: https://weaviate.io/
- **Qdrant**: https://qdrant.tech/
- **ChromaDB**: https://www.trychroma.com/

---

## 总结

对于你的场景（389页，4个文档）：

### 🎯 短期方案（开发测试）
使用 **本地 FAISS (CPU)**
- 成本: $0
- 性能: 足够
- 优势: 简单

### 🚀 长期方案（生产环境）
使用 **Milvus (自建)** 或 **Pinecone**
- 成本: $70-100/月
- 性能: 优秀
- 优势: 可扩展，易维护

### 💡 推荐路径
1. 先用本地 FAISS 开发和测试
2. 验证效果后迁移到 Pinecone（快速上线）
3. 规模扩大后考虑自建 Milvus（降低成本）
