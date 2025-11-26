#!/usr/bin/env python3
"""
多路召回系统测试脚本
用于快速验证系统功能
"""

import sys
import time
import logging
from pathlib import Path

# 添加当前目录到路径
sys.path.append(str(Path(__file__).parent))

from multimodal_multipath_retrieval import MultiPathRetriever, load_pdf_images
from multipath_config import get_recommended_config, print_config_summary

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_basic_functionality():
    """测试基本功能"""
    logger.info("开始基本功能测试")
    
    # 配置
    DEVICE = "cuda:6"
    PDF_PATH = "contents/2024_Tencent_ESG.pdf"
    HF_TOKEN = "***REMOVED***"
    
    # 测试查询
    test_queries = [
        "2022年员工总数是多少？",
        "温室气体排放情况"
    ]
    
    try:
        # 1. 检查文件存在
        if not Path(PDF_PATH).exists():
            logger.error(f"PDF文件不存在: {PDF_PATH}")
            return False
            
        # 2. 加载PDF (只加载前10页用于测试)
        logger.info("加载PDF文档...")
        images = load_pdf_images(PDF_PATH)
        test_images = images[:10]  # 只用前10页测试
        logger.info(f"测试文档页数: {len(test_images)}")
        
        # 3. 获取推荐配置
        config = get_recommended_config(len(test_images), DEVICE)
        print_config_summary(config)
        
        # 4. 初始化系统
        logger.info("初始化多路召回系统...")
        retriever = MultiPathRetriever(device=DEVICE)
        
        # 5. 加载模型
        logger.info("加载模型...")
        retriever.load_models(metaclip_token=HF_TOKEN)
        
        # 6. 构建索引
        logger.info("构建索引...")
        retriever.build_indices(
            images=test_images,
            metaclip_index_path="./test_indices/metaclip_test.faiss",
            nvidia_index_dir="./test_indices/nvidia_test"
        )
        
        # 7. 执行检索测试
        logger.info("执行检索测试...")
        for i, query in enumerate(test_queries, 1):
            logger.info(f"测试查询 {i}: {query}")
            
            start_time = time.time()
            results = retriever.retrieve(
                query=query,
                top_k_per_path=10,
                final_top_k=5,
                fusion_method="weighted_sum"
            )
            elapsed = time.time() - start_time
            
            logger.info(f"检索完成，耗时: {elapsed:.3f}秒")
            logger.info(f"返回结果数: {len(results)}")
            
            # 显示前3个结果
            for rank, result in enumerate(results[:3], 1):
                logger.info(f"  {rank}. 第{result.page_num}页 (分数: {result.final_score:.4f})")
        
        logger.info("? 基本功能测试通过")
        return True
        
    except Exception as e:
        logger.error(f"? 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_system():
    """测试配置系统"""
    logger.info("开始配置系统测试")
    
    try:
        # 测试不同规模的推荐配置
        test_cases = [50, 500, 5000]
        
        for num_pages in test_cases:
            config = get_recommended_config(num_pages)
            logger.info(f"? {num_pages}页文档配置生成成功")
            
            # 验证配置合理性
            assert config.retrieval.top_k_per_path > 0
            assert config.retrieval.final_top_k > 0
            assert config.fusion.metaclip_weight + config.fusion.nvidia_weight == 1.0
        
        logger.info("? 配置系统测试通过")
        return True
        
    except Exception as e:
        logger.error(f"? 配置系统测试失败: {e}")
        return False

def main():
    """主测试函数"""
    logger.info("=" * 60)
    logger.info("多路召回系统测试开始")
    logger.info("=" * 60)
    
    test_results = []
    
    # 测试1: 配置系统
    logger.info("\n? 测试1: 配置系统")
    test_results.append(("配置系统", test_config_system()))
    
    # 测试2: 基本功能 (需要GPU和模型)
    logger.info("\n? 测试3: 基本功能")
    if input("是否执行完整功能测试? (需要GPU和网络) [y/N]: ").lower() == 'y':
        test_results.append(("基本功能", test_basic_functionality()))
    else:
        logger.info("跳过基本功能测试")
        test_results.append(("基本功能", "跳过"))
    
    # 汇总结果
    logger.info("\n" + "=" * 60)
    logger.info("测试结果汇总")
    logger.info("=" * 60)
    
    passed = 0
    total = 0
    
    for test_name, result in test_results:
        if result == "跳过":
            logger.info(f"??  {test_name}: 跳过")
        elif result:
            logger.info(f"? {test_name}: 通过")
            passed += 1
            total += 1
        else:
            logger.info(f"? {test_name}: 失败")
            total += 1
    
    if total > 0:
        logger.info(f"\n? 测试通过率: {passed}/{total} ({passed/total*100:.1f}%)")
    
    logger.info("测试完成")

if __name__ == "__main__":
    main()
