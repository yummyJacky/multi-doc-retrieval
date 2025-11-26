# 多模态多路召回系统

## 概述

本系统融合了MetaCLIP2和Nvidia FAISS两种检索方法，实现了多路召回和智能结果融合，提供了更全面、更准确的文档检索能力。

## 系统架构

```
查询输入
    ├── MetaCLIP2路径 (图像-文本跨模态检索)
    │   ├── 文本编码
    │   ├── 图像索引搜索
    │   └── 相似度计算
    │
    ├── Nvidia FAISS路径 (语义文档检索)
    │   ├── 查询编码
    │   ├── FAISS索引搜索
    │   └── 语义匹配
    │
    └── 结果融合模块
        ├── 加权求和
        ├── 最大值融合
        ├── RRF融合
        └── 最终排序
```

## 核心文件

### 1. `multimodal_multipath_retrieval.py`
主要实现文件，包含：
- `MetaCLIPRetriever`: MetaCLIP2图像检索器
- `NvidiaFAISSRetriever`: Nvidia FAISS文档检索器  
- `MultiPathRetriever`: 多路召回系统主类
- 结果融合算法

### 2. `multipath_config.py`
配置管理文件，包含：
- 各种预定义配置
- 配置类定义
- 推荐配置生成器

### 3. `multimodal_multipath_demo.ipynb`
完整使用演示，包含：
- 环境设置
- 模型加载
- 索引构建
- 检索演示
- 性能分析

## 主要特性

### ? 多路召回
- **MetaCLIP2路径**: 擅长理解图像中的视觉元素、图表、文字等
- **Nvidia FAISS路径**: 擅长语义理解、文档结构分析、上下文关联

### ? 高效索引
- **FAISS加速**: 支持百万级文档快速检索
- **GPU优化**: 充分利用GPU计算资源
- **多种索引类型**: Flat、IVF、IVF-PQ等适应不同规模

### ? 智能融合
- **加权求和**: 平衡两种检索方式的优势
- **最大值融合**: 突出最相关的结果
- **RRF融合**: 基于排名的融合，减少分数偏差

### ? 灵活配置
- **预定义配置**: 针对不同场景的优化配置
- **参数调优**: 支持权重、阈值等参数调整
- **自动推荐**: 根据文档规模自动推荐配置

## 快速开始

### 1. 环境准备
```bash
pip install torch transformers faiss-gpu pillow pdf2image
```

### 2. 基本使用
```python
from multimodal_multipath_retrieval import MultiPathRetriever, load_pdf_images

# 加载文档
images = load_pdf_images("your_document.pdf")

# 初始化系统
retriever = MultiPathRetriever(device="cuda:0")

# 加载模型
retriever.load_models(metaclip_token="your_hf_token")

# 构建索引
retriever.build_indices(images=images)

# 执行检索
results = retriever.retrieve(
    query="你的查询",
    top_k_per_path=20,
    final_top_k=10,
    fusion_method="weighted_sum"
)
```

### 3. 配置使用
```python
from multipath_config import get_recommended_config, print_config_summary

# 获取推荐配置
config = get_recommended_config(num_pages=len(images))
print_config_summary(config)

# 使用配置
retriever = MultiPathRetriever(device=config.metaclip.device)
```

## 融合策略

### 1. 加权求和 (weighted_sum)
```
final_score = w1 * metaclip_score + w2 * nvidia_score
```
- **适用场景**: 通用场景，需要平衡视觉和文本检索
- **参数**: metaclip_weight, nvidia_weight

### 2. 最大值融合 (max)
```
final_score = max(metaclip_score, nvidia_score)
```
- **适用场景**: 需要高精度，突出最相关结果
- **参数**: 无

### 3. RRF融合 (rrf)
```
final_score = 1/(k + rank1) + 1/(k + rank2)
```
- **适用场景**: 不同检索器分数分布差异较大
- **参数**: rrf_k (通常设为60)

## 索引类型选择

| 索引类型 | 适用规模 | 速度 | 精度 | 内存占用 |
|---------|---------|------|------|----------|
| Flat | <10K页 | 慢 | 100% | 高 |
| IVFFlat | 10K-1M页 | 中等 | 95-99% | 中等 |
| IVFPQ | >1M页 | 快 | 90-95% | 低 |

## 性能优化建议

### 1. 硬件配置
- **GPU内存**: 建议16GB+用于大模型加载
- **系统内存**: 建议32GB+用于大文档处理
- **存储**: SSD用于索引快速读写

### 2. 参数调优
- **批处理大小**: 根据GPU内存调整batch_size
- **检索数量**: 平衡检索质量和速度
- **融合权重**: 根据应用场景调整权重比例

### 3. 索引优化
- **预构建索引**: 离线构建索引，在线快速加载
- **索引压缩**: 大规模文档使用IVFPQ索引
- **分片策略**: 超大文档集合考虑分片索引

## 应用场景

### 1. 企业文档检索
- 年报、财报等复杂文档
- 包含图表、表格的技术文档
- 多语言文档检索

### 2. 学术研究
- 论文检索和分析
- 实验数据查找
- 文献综述辅助

### 3. 法律文档
- 合同条款检索
- 法规条文查找
- 案例分析

## 扩展方向

### 1. 更多检索路径
- OCR文字识别路径
- 表格结构化检索
- 图表专门识别

### 2. 智能优化
- 自适应融合策略
- 查询意图识别
- 个性化权重学习

### 3. 多文档支持
- 跨文档检索
- 文档关联分析
- 知识图谱构建

## 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减少batch_size
   - 使用CPU版本FAISS
   - 分批处理文档

2. **模型加载失败**
   - 检查HuggingFace token
   - 确认网络连接
   - 使用镜像源

3. **检索速度慢**
   - 使用更快的索引类型
   - 减少top_k数量
   - 启用GPU加速

4. **检索质量差**
   - 调整融合权重
   - 尝试不同融合策略
   - 增加检索路径数量

### 日志分析
系统提供详细的日志信息，可以通过日志分析性能瓶颈和错误原因。

## 贡献指南

欢迎提交Issue和Pull Request来改进系统：

1. 性能优化
2. 新的融合策略
3. 更多检索路径
4. 文档和示例改进

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 更新日志

### v1.0.0 (2024-11-18)
- 初始版本发布
- 支持MetaCLIP2和Nvidia FAISS双路召回
- 实现三种融合策略
- 提供完整的配置管理
- 包含详细的使用示例
