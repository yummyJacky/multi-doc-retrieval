# Milvus 快速启动指南

## 🚀 5分钟上手 Milvus

### Step 1: 安装 Milvus（选择一种方式）

#### 方式1: Docker（推荐）

```bash
# 拉取镜像
docker pull milvusdb/milvus:latest

# 启动 Milvus
docker run -d --name milvus_standalone \
  -p 19530:19530 \
  -p 9091:9091 \
  -v $(pwd)/milvus_data:/var/lib/milvus \
  milvusdb/milvus:latest

# 检查状态
docker ps | grep milvus
```

#### 方式2: Docker Compose（推荐用于生产）

```bash
# 下载配置文件
wget https://github.com/milvus-io/milvus/releases/download/v2.3.0/milvus-standalone-docker-compose.yml -O docker-compose.yml

# 启动
docker-compose up -d

# 检查状态
docker-compose ps
```

#### 方式3: 使用 Zilliz Cloud（云服务）

1. 访问 https://cloud.zilliz.com/
2. 注册并创建免费集群
3. 获取连接信息（host, port, token）

### Step 2: 安装 Python 客户端

```bash
pip install pymilvus
```

### Step 3: 测试连接

```python
from pymilvus import connections, utility

# 连接到 Milvus
connections.connect(host="localhost", port="19530")

# 检查版本
print(f"Milvus 版本: {utility.get_server_version()}")

# 输出: Milvus 版本: v2.3.0
```

---

## 📦 迁移现有项目

### 方式1: 使用集成类（推荐）

```python
# 使用 MilvusRAGPipeline 替代 NvidiaRAGPipeline
from examples.milvus_integration_example import MilvusRAGPipeline

# 初始化（其他代码不变）
rag_pipeline = MilvusRAGPipeline(
    retriever_model=retriever_model,
    milvus_host="localhost",
    milvus_port=19530,
    collection_name="my_documents",
    device=9
)

# 编码文档（API 完全相同）
rag_pipeline.encode_documents(docid2images)

# 构建索引（自动上传到 Milvus）
rag_pipeline.build_unified_index()

# 检索（API 完全相同）
results = rag_pipeline.retrieve_multi_doc(
    queries=["问题"],
    top_k=10
)
```

### 方式2: 手动集成

```python
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType

# 1. 连接
connections.connect(host="localhost", port="19530")

# 2. 创建集合
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="page_id", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=3072),
]
schema = CollectionSchema(fields=fields)
collection = Collection(name="documents", schema=schema)

# 3. 创建索引
index_params = {
    "metric_type": "IP",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 128}
}
collection.create_index(field_name="embedding", index_params=index_params)

# 4. 插入数据
# embeddings: (n_vectors, dim)
# page_ids: List[str]
collection.insert([page_ids, embeddings.tolist()])
collection.flush()

# 5. 搜索
collection.load()
results = collection.search(
    data=query_embedding.tolist(),
    anns_field="embedding",
    param={"metric_type": "IP", "params": {"nprobe": 10}},
    limit=10,
    output_fields=["page_id"]
)
```

---

## 🎯 完整示例

### 运行示例代码

```bash
cd examples
python milvus_integration_example.py
```

### 预期输出

```
================================================================================
Milvus 云向量数据库集成示例
================================================================================

1. 加载 Retriever 模型...
✓ 模型加载完成

2. 初始化 Milvus RAG Pipeline...
✓ 已连接到 Milvus: localhost:19530
✓ 创建新集合: multi_doc_demo
✓ 创建索引完成
✓ Pipeline 初始化完成

3. 加载文档...
  ✓ tencent_esg: 10 页
  ✓ archi_esg: 10 页

4. 编码文档...
[编码进度...]

5. 构建索引并上传到 Milvus...
[上传进度...]
✓ 上传完成！
  - 集合名称: multi_doc_demo
  - 总向量数: 36040

6. 检索测试...
[检索结果...]
```

---

## 🔧 常见问题

### Q1: 连接失败怎么办？

```python
# 检查 Milvus 是否运行
docker ps | grep milvus

# 查看日志
docker logs milvus_standalone

# 重启 Milvus
docker restart milvus_standalone
```

### Q2: 如何查看已有的集合？

```python
from pymilvus import connections, utility

connections.connect(host="localhost", port="19530")
print(utility.list_collections())
```

### Q3: 如何删除集合？

```python
from pymilvus import utility

utility.drop_collection("collection_name")
```

### Q4: 如何查看集合信息？

```python
from pymilvus import Collection

collection = Collection("collection_name")
print(f"向量数量: {collection.num_entities}")
print(f"Schema: {collection.schema}")
```

### Q5: 如何备份数据？

```bash
# 备份 Milvus 数据目录
docker exec milvus_standalone tar -czf /tmp/milvus_backup.tar.gz /var/lib/milvus

# 复制到本地
docker cp milvus_standalone:/tmp/milvus_backup.tar.gz ./
```

---

## 📊 性能优化

### 1. 索引类型选择

```python
# IVF_FLAT: 平衡速度和精度（推荐）
index_params = {
    "metric_type": "IP",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 128}
}

# IVF_PQ: 更快，略微损失精度
index_params = {
    "metric_type": "IP",
    "index_type": "IVF_PQ",
    "params": {"nlist": 128, "m": 8}
}

# HNSW: 最快，内存占用大
index_params = {
    "metric_type": "IP",
    "index_type": "HNSW",
    "params": {"M": 16, "efConstruction": 200}
}
```

### 2. 搜索参数调优

```python
# nprobe: 探测的聚类数量（越大越精确，越慢）
search_params = {"metric_type": "IP", "params": {"nprobe": 10}}

# 推荐值:
# - 快速搜索: nprobe=5
# - 平衡: nprobe=10
# - 高精度: nprobe=20
```

### 3. 批量操作

```python
# 批量插入（更快）
batch_size = 1000
for i in range(0, len(data), batch_size):
    batch = data[i:i+batch_size]
    collection.insert(batch)

collection.flush()  # 确保数据持久化
```

---

## 🌐 使用云服务（Zilliz Cloud）

### 1. 注册并创建集群

1. 访问 https://cloud.zilliz.com/
2. 注册账号（免费）
3. 创建集群（选择免费层）
4. 获取连接信息

### 2. 连接到云集群

```python
from pymilvus import connections

connections.connect(
    alias="default",
    host="your-cluster.aws-us-west-2.vectordb.zillizcloud.com",
    port="19530",
    user="username",
    password="password",
    secure=True
)
```

### 3. 修改示例代码

```python
# 只需修改连接参数
rag_pipeline = MilvusRAGPipeline(
    retriever_model=retriever_model,
    milvus_host="your-cluster.aws-us-west-2.vectordb.zillizcloud.com",
    milvus_port=19530,
    collection_name="my_documents",
    device=9
)
```

---

## 📈 监控和管理

### 1. Web UI（Attu）

```bash
# 安装 Attu（Milvus 的 Web UI）
docker run -d --name attu \
  -p 3000:3000 \
  -e MILVUS_URL=localhost:19530 \
  zilliz/attu:latest

# 访问: http://localhost:3000
```

### 2. 命令行工具

```python
from pymilvus import connections, utility, Collection

# 连接
connections.connect(host="localhost", port="19530")

# 查看所有集合
print("所有集合:", utility.list_collections())

# 查看集合详情
collection = Collection("collection_name")
print(f"向量数量: {collection.num_entities}")
print(f"索引信息: {collection.index().params}")

# 查看统计信息
stats = collection.get_stats()
print(f"统计信息: {stats}")
```

---

## 💡 最佳实践

### 1. 开发环境

```python
# 使用本地 Docker
docker run -d --name milvus_dev \
  -p 19530:19530 \
  milvusdb/milvus:latest
```

### 2. 生产环境

```python
# 使用 Zilliz Cloud 或自建集群
# 配置备份和监控
# 使用连接池
```

### 3. 数据管理

```python
# 定期备份
# 监控索引大小
# 清理过期数据
```

---

## 🔗 相关资源

- **Milvus 官网**: https://milvus.io/
- **Milvus 文档**: https://milvus.io/docs
- **Zilliz Cloud**: https://cloud.zilliz.com/
- **GitHub**: https://github.com/milvus-io/milvus
- **社区**: https://discuss.milvus.io/

---

## 总结

使用 Milvus 的优势：

✅ **简单**: 5分钟即可启动
✅ **可靠**: 自动持久化，不会丢失数据
✅ **可扩展**: 支持从 GB 到 PB 级数据
✅ **高性能**: 毫秒级检索响应
✅ **灵活**: 支持多种索引类型和搜索策略

对于你的场景（389页，700K向量），Milvus 是完美的选择！
