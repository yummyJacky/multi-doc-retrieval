"""
多路召回系统配置文件
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class MetaCLIPConfig:
    """MetaCLIP配置"""
    model_name: str = "facebook/metaclip-2-worldwide-giant"
    embedding_dim: int = 1280
    device: str = "cuda:0"
    dtype: str = "float32"
    attn_implementation: str = "sdpa"

@dataclass
class NvidiaConfig:
    """Nvidia配置"""
    model_name: str = "nvidia/llama-nemoretriever-colembed-3b-v1"
    embedding_dim: int = 3072
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"
    revision: str = "50c36f4d5271c6851aa08bd26d69f6e7ca8b870c"

@dataclass
class FAISSConfig:
    """FAISS索引配置"""
    index_type: str = "ivfflat"  # "flat", "ivfflat", "ivfpq"
    use_gpu: bool = True
    nlist: Optional[int] = None  # 自动计算为sqrt(n_docs)
    nprobe: int = 10  # 搜索时的聚类数
    
    # IVF-PQ特定参数
    m: int = 64  # PQ子向量数量
    nbits: int = 8  # 每个子向量的位数

@dataclass
class FusionConfig:
    """融合配置"""
    method: str = "weighted_sum"  # "weighted_sum", "max", "rrf"
    metaclip_weight: float = 0.4
    nvidia_weight: float = 0.6
    rrf_k: int = 60  # RRF参数
    
@dataclass
class RetrievalConfig:
    """检索配置"""
    top_k_per_path: int = 20
    final_top_k: int = 10
    batch_size: int = 8
    
@dataclass
class SystemConfig:
    """系统总配置"""
    metaclip: MetaCLIPConfig
    nvidia: NvidiaConfig
    faiss: FAISSConfig
    fusion: FusionConfig
    retrieval: RetrievalConfig
    
    # 文件路径
    hf_token: Optional[str] = None
    pdf_path: Optional[str] = None
    index_save_dir: str = "./indices"
    
    # 日志配置
    log_level: str = "INFO"
    
    @classmethod
    def default(cls, device: str = "cuda:0") -> "SystemConfig":
        """创建默认配置"""
        return cls(
            metaclip=MetaCLIPConfig(device=device),
            nvidia=NvidiaConfig(device=device),
            faiss=FAISSConfig(),
            fusion=FusionConfig(),
            retrieval=RetrievalConfig()
        )
    
    @classmethod
    def for_large_docs(cls, device: str = "cuda:0") -> "SystemConfig":
        """大文档优化配置"""
        config = cls.default(device)
        config.faiss.index_type = "ivfpq"
        config.retrieval.top_k_per_path = 50
        config.retrieval.final_top_k = 20
        return config
    
    @classmethod
    def for_fast_retrieval(cls, device: str = "cuda:0") -> "SystemConfig":
        """快速检索配置"""
        config = cls.default(device)
        config.faiss.index_type = "flat"
        config.retrieval.top_k_per_path = 10
        config.retrieval.final_top_k = 5
        return config

# 预定义配置
DEFAULT_CONFIG = SystemConfig.default()
LARGE_DOCS_CONFIG = SystemConfig.for_large_docs()
FAST_RETRIEVAL_CONFIG = SystemConfig.for_fast_retrieval()

# 融合策略说明
FUSION_STRATEGIES = {
    "weighted_sum": {
        "description": "加权求和，平衡两种检索方式",
        "best_for": "通用场景，需要平衡视觉和文本检索",
        "params": ["metaclip_weight", "nvidia_weight"]
    },
    "max": {
        "description": "取最大分数，突出最相关结果",
        "best_for": "需要高精度的场景",
        "params": []
    },
    "rrf": {
        "description": "基于排名的融合，减少分数偏差",
        "best_for": "不同检索器分数分布差异较大时",
        "params": ["rrf_k"]
    }
}

# 索引类型说明
INDEX_TYPES = {
    "flat": {
        "description": "精确搜索，无压缩",
        "best_for": "小规模文档(<10K页)",
        "speed": "慢",
        "accuracy": "100%"
    },
    "ivfflat": {
        "description": "聚类索引，平衡速度和精度",
        "best_for": "中等规模文档(10K-1M页)",
        "speed": "中等",
        "accuracy": "95-99%"
    },
    "ivfpq": {
        "description": "聚类+量化，最快速度",
        "best_for": "大规模文档(>1M页)",
        "speed": "快",
        "accuracy": "90-95%"
    }
}

def get_recommended_config(num_pages: int, device: str = "cuda:0") -> SystemConfig:
    """根据文档页数推荐配置"""
    if num_pages < 100:
        return SystemConfig.for_fast_retrieval(device)
    elif num_pages < 1000:
        return SystemConfig.default(device)
    else:
        return SystemConfig.for_large_docs(device)

def print_config_summary(config: SystemConfig):
    """打印配置摘要"""
    print("? 系统配置摘要")
    print("=" * 50)
    print(f"MetaCLIP模型: {config.metaclip.model_name}")
    print(f"Nvidia模型:  {config.nvidia.model_name}")
    print(f"设备:        {config.metaclip.device}")
    print(f"FAISS索引:   {config.faiss.index_type}")
    print(f"融合方法:    {config.fusion.method}")
    print(f"融合权重:    MetaCLIP={config.fusion.metaclip_weight}, Nvidia={config.fusion.nvidia_weight}")
    print(f"检索参数:    每路径{config.retrieval.top_k_per_path}, 最终{config.retrieval.final_top_k}")
    print("=" * 50)
