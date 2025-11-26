# 多模态RAG评估系统

基于deepeval框架构建的多模态RAG评估系统，专门用于评估Nvidia FAISS MultiDoc RAG pipeline的性能。

## 功能特性

### 🎯 核心评估指标

基于deepeval的多模态指标，支持以下评估维度：

1. **Answer Relevancy (答案相关性)**
   - 评估生成答案与输入查询的相关程度
   - 支持文本和图像的多模态输入

2. **Faithfulness (忠实性)**
   - 评估生成答案是否忠实于检索到的上下文
   - 检测幻觉和事实错误

3. **Contextual Relevancy (上下文相关性)**
   - 评估检索上下文与查询的相关程度
   - 衡量检索器的质量

4. **Contextual Precision (上下文精确度)**
   - 评估相关文档在检索结果中的排序质量
   - 需要期望输出作为参考

5. **Contextual Recall (上下文召回率)**
   - 评估检索系统捕获相关信息的完整性
   - 需要期望输出作为参考

### 🔧 技术架构

- **检索器**: Nvidia NemoRetriever (llama-nemoretriever-colembed-3b-v1)
- **重排序**: MonoVLM
- **问答模型**: Doubao Vision API
- **索引**: FAISS (IVFFlat)
- **评估框架**: DeepEval

## 文件结构

```
├── multimodal_rag_evaluation.py      # 完整的评估系统
├── simple_multimodal_evaluation.py   # 简化版评估脚本
├── Nvidia_FAISS_MultiDoc.ipynb      # 原始RAG实现
├── README_evaluation.md             # 本文档
└── evaluation_results/              # 评估结果目录
```

## 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install deepeval torch transformers pdf2image faiss-cpu numpy

# 设置API密钥（如果使用OpenAI模型评估）
export OPENAI_API_KEY="your-api-key"

# 设置Confident AI API密钥（可选，用于结果可视化）
export CONFIDENT_API_KEY="your-confident-api-key"
```

### 2. 准备文档

将PDF文档放置在 `contents/` 目录下：

```
contents/
├── 2024_Tencent_ESG.pdf
├── 2024_architecture_ESG.pdf
├── 2024_sanqi_ESG.pdf
└── 2024_zhongxing_ESG.pdf
```

### 3. 运行简化评估

```python
from simple_multimodal_evaluation import SimpleMultimodalEvaluator

# 配置文档
documents = {
    "tencent_esg": "contents/2024_Tencent_ESG.pdf",
    "archi_esg": "contents/2024_architecture_ESG.pdf",
    "sanqi_esg": "contents/2024_sanqi_ESG.pdf",
    "zhongxing_esg": "contents/2024_zhongxing_ESG.pdf"
}

# 测试查询
queries = [
    "针对GRI403-3，各个公司有什么区别？",
    "各公司在环境保护方面的主要措施是什么？"
]

# 运行评估
evaluator = SimpleMultimodalEvaluator(device=9)
results = evaluator.run_batch_evaluation(documents, queries)
```

### 4. 运行完整评估

```python
from multimodal_rag_evaluation import MultimodalRAGEvaluator

evaluator = MultimodalRAGEvaluator(
    device=9,
    model_name="gpt-4o",
    threshold=0.7,
    include_reason=True
)

results = evaluator.run_comprehensive_evaluation(
    documents=documents,
    queries=queries,
    expected_outputs=expected_outputs,  # 可选
    save_index=True
)
```

## 评估流程

### 1. 系统初始化
- 加载Nvidia NemoRetriever模型
- 初始化FAISS索引
- 加载重排序和VQA模型

### 2. 文档处理
- PDF转图像 (200 DPI)
- 文档编码 (3072维向量)
- 构建统一索引

### 3. 查询处理
- 多文档检索 (Token级MaxSim)
- 图像重排序 (MonoVLM)
- 答案生成 (VQA模型)

### 4. 评估执行
- 创建MLLMTestCase
- 运行多模态指标评估
- 生成详细报告

## 评估结果

### 结果格式

```json
{
  "evaluation_info": {
    "timestamp": "2024-01-01T12:00:00",
    "model_name": "gpt-4o",
    "num_documents": 4,
    "num_queries": 2
  },
  "query_results": [
    {
      "query": "针对GRI403-3，各个公司有什么区别？",
      "actual_output": "生成的答案...",
      "metrics": {
        "answer_relevancy": {
          "score": 0.85,
          "reason": "答案与查询高度相关...",
          "passed": true
        },
        "faithfulness": {
          "score": 0.92,
          "reason": "答案忠实于检索上下文...",
          "passed": true
        }
      }
    }
  ],
  "summary": {
    "answer_relevancy": {
      "average_score": 0.85,
      "pass_rate": 0.8
    }
  }
}
```

### 指标解释

- **分数范围**: 0.0 - 1.0 (越高越好)
- **通过阈值**: 默认0.7 (可配置)
- **通过率**: 超过阈值的测试用例比例

## 配置选项

### 评估器配置

```python
evaluator = MultimodalRAGEvaluator(
    device=9,                    # GPU设备
    model_name="gpt-4o",        # 评估模型
    threshold=0.7,              # 通过阈值
    include_reason=True,        # 包含评估原因
    verbose_mode=False          # 详细模式
)
```

### RAG配置

```python
faiss_config = {
    "embedding_dim": 3072,      # 嵌入维度
    "index_type": "ivfflat",    # 索引类型
    "nlist": 20,                # 聚类中心数
    "use_gpu": False            # 使用GPU索引
}
```

## 性能优化

### 内存管理
- 使用`use_gpu=False`避免大索引保存问题
- 批量处理减少内存碎片
- 及时清理中间结果

### 速度优化
- IVFFlat索引比Flat索引快3倍
- 批量编码和评估
- 异步模式处理

### GPU使用
- 自动内存监控
- 设备间负载均衡
- Flash Attention加速

## 故障排除

### 常见问题

1. **内存不足**
   ```
   RuntimeError: CUDA out of memory
   ```
   - 减少batch_size
   - 使用CPU索引
   - 清理GPU缓存

2. **索引保存失败**
   ```
   RuntimeError: cannot create std::vector larger than max_size()
   ```
   - 设置`save_dir=None`
   - 使用CPU索引
   - 分批保存

3. **模型加载失败**
   ```
   OSError: Can't load tokenizer
   ```
   - 检查网络连接
   - 设置HF_ENDPOINT镜像
   - 验证模型名称

### 批量评估

```python
# 从文件加载查询
with open('queries.txt', 'r') as f:
    queries = [line.strip() for line in f]

# 并行评估
results = evaluator.run_batch_evaluation(documents, queries)
```
## 最佳实践

1. **查询设计**: 使用多样化、具有挑战性的查询
2. **文档准备**: 确保PDF质量和格式一致性
3. **阈值设置**: 根据业务需求调整通过阈值
4. **批量处理**: 使用批量模式提高效率
5. **结果分析**: 结合定量和定性分析
6. **持续优化**: 基于评估结果迭代改进系统

## 参考资料

- [DeepEval Documentation](https://docs.deepeval.com/)
- [Nvidia NemoRetriever](https://huggingface.co/nvidia/llama-nemoretriever-colembed-3b-v1)
- [M3DocRAG Paper](https://arxiv.org/abs/2401.02103)
- [FAISS Documentation](https://faiss.ai/)
