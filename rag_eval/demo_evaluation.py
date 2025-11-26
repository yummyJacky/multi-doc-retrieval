#!/usr/bin/env python3
"""
多模态RAG评估演示脚本

这个脚本演示如何使用评估系统对多文档RAG进行评估。
包含完整的端到端示例。
"""

import os
import json
from multimodal_evaluation import SimpleMultimodalEvaluator


def demo_single_query_evaluation():
    """演示单个查询的评估过程"""
    print("="*80)
    print("演示：单个查询评估")
    print("="*80)
    
    # 检查文档是否存在
    documents = {
        "tencent_esg": "contents/2024_Tencent_ESG.pdf",
        "archi_esg": "contents/2024_architecture_ESG.pdf",
        "sanqi_esg": "contents/2024_sanqi_ESG.pdf",
        "zhongxing_esg": "contents/2024_zhongxing_ESG.pdf"
    }
    
    missing_docs = []
    for doc_id, path in documents.items():
        if not os.path.exists(path):
            missing_docs.append(path)
    
    if missing_docs:
        print("⚠️  以下文档文件未找到:")
        for doc in missing_docs:
            print(f"   - {doc}")
        print("\n请确保PDF文件存在于contents/目录下")
        return
    
    # 初始化评估器
    evaluator = SimpleMultimodalEvaluator(device=9)
    
    try:
        # 设置RAG系统
        print("🔧 设置RAG系统...")
        evaluator.setup_rag_system()
        
        # 加载文档
        print("📚 加载文档...")
        evaluator.load_documents(documents)
        
        # 单个查询评估
        query = "针对GRI403-3，各个公司有什么区别？"
        print(f"❓ 评估查询: {query}")
        
        result = evaluator.evaluate_query(query, model_name="gpt-4o")
        
        # 显示结果
        print("\n📊 评估结果:")
        print(f"查询: {result['query']}")
        print(f"生成答案长度: {len(result.get('actual_output', ''))}")
        print(f"检索上下文数量: {result.get('retrieval_context_count', 0)}")
        
        print("\n📈 指标分数:")
        for metric_name, metric_data in result.get('metrics', {}).items():
            if metric_data.get('score') is not None:
                score = metric_data['score']
                passed = "✅" if metric_data.get('passed', False) else "❌"
                print(f"  {metric_name}: {score:.4f} {passed}")
                if metric_data.get('reason'):
                    print(f"    原因: {metric_data['reason'][:100]}...")
        
        print("\n✅ 单个查询评估完成")
        
    except Exception as e:
        print(f"❌ 评估失败: {str(e)}")


def demo_batch_evaluation():
    """演示批量评估"""
    print("\n" + "="*80)
    print("演示：批量评估")
    print("="*80)
    
    # 配置
    documents = {
        "tencent_esg": "contents/2024_Tencent_ESG.pdf",
        "archi_esg": "contents/2024_architecture_ESG.pdf",
        "sanqi_esg": "contents/2024_sanqi_ESG.pdf",
        "zhongxing_esg": "contents/2024_zhongxing_ESG.pdf"
    }
    
    queries = [
        "针对GRI403-3，各个公司有什么区别？",
        "各公司在环境保护方面的主要措施是什么？",
        "社会责任报告中提到的员工培训情况如何？"
    ]
    
    # 检查文档
    missing_docs = [path for path in documents.values() if not os.path.exists(path)]
    if missing_docs:
        print("⚠️  部分文档文件未找到，跳过批量评估演示")
        return
    
    # 初始化评估器
    evaluator = SimpleMultimodalEvaluator(device=9)
    
    try:
        print(f"🚀 开始批量评估 {len(queries)} 个查询...")
        
        # 运行批量评估
        results = evaluator.run_batch_evaluation(
            documents=documents,
            queries=queries,
            model_name="gpt-4o"
        )
        
        # 显示汇总结果
        print("\n📊 批量评估结果汇总:")
        print(f"文档数量: {results['evaluation_info']['num_documents']}")
        print(f"查询数量: {results['evaluation_info']['num_queries']}")
        
        if 'summary' in results:
            print("\n📈 指标汇总:")
            for metric_name, summary in results['summary'].items():
                print(f"  {metric_name}:")
                print(f"    平均分数: {summary['average_score']:.4f}")
                print(f"    分数范围: {summary['min_score']:.4f} - {summary['max_score']:.4f}")
                print(f"    通过率: {summary['pass_rate']:.2%}")
        
        # 显示每个查询的结果
        print("\n📝 各查询详细结果:")
        for i, query_result in enumerate(results['query_results'], 1):
            print(f"\n  查询 {i}: {query_result['query'][:50]}...")
            if 'metrics' in query_result:
                for metric_name, metric_data in query_result['metrics'].items():
                    if metric_data.get('score') is not None:
                        score = metric_data['score']
                        passed = "✅" if metric_data.get('passed', False) else "❌"
                        print(f"    {metric_name}: {score:.4f} {passed}")
        
        print("\n✅ 批量评估完成")
        
        # 保存结果文件路径
        result_files = [f for f in os.listdir('evaluation_results') if f.startswith('evaluation_')]
        if result_files:
            latest_file = max(result_files)
            print(f"📁 详细结果已保存到: evaluation_results/{latest_file}")
        
    except Exception as e:
        print(f"❌ 批量评估失败: {str(e)}")


def demo_result_analysis():
    """演示结果分析"""
    print("\n" + "="*80)
    print("演示：结果分析")
    print("="*80)
    
    # 查找最新的评估结果文件
    if not os.path.exists('evaluation_results'):
        print("⚠️  未找到评估结果目录，请先运行评估")
        return
    
    result_files = [f for f in os.listdir('evaluation_results') if f.endswith('.json')]
    if not result_files:
        print("⚠️  未找到评估结果文件，请先运行评估")
        return
    
    # 加载最新结果
    latest_file = max(result_files)
    result_path = os.path.join('evaluation_results', latest_file)
    
    try:
        with open(result_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        print(f"📊 分析结果文件: {latest_file}")
        
        # 基本统计
        num_queries = len(results.get('query_results', []))
        print(f"查询数量: {num_queries}")
        
        if 'summary' in results:
            print("\n📈 性能分析:")
            
            # 找出表现最好和最差的指标
            metric_scores = {}
            for metric_name, summary in results['summary'].items():
                metric_scores[metric_name] = summary['average_score']
            
            if metric_scores:
                best_metric = max(metric_scores, key=metric_scores.get)
                worst_metric = min(metric_scores, key=metric_scores.get)
                
                print(f"🏆 表现最好的指标: {best_metric} ({metric_scores[best_metric]:.4f})")
                print(f"⚠️  需要改进的指标: {worst_metric} ({metric_scores[worst_metric]:.4f})")
        
        # 查询级别分析
        if 'query_results' in results:
            print("\n🔍 查询级别分析:")
            
            query_scores = []
            for i, query_result in enumerate(results['query_results']):
                if 'metrics' in query_result:
                    scores = [
                        metric_data.get('score', 0) 
                        for metric_data in query_result['metrics'].values()
                        if metric_data.get('score') is not None
                    ]
                    if scores:
                        avg_score = sum(scores) / len(scores)
                        query_scores.append((i, avg_score, query_result['query']))
            
            if query_scores:
                # 排序找出最好和最差的查询
                query_scores.sort(key=lambda x: x[1], reverse=True)
                
                print(f"🎯 表现最好的查询 (分数: {query_scores[0][1]:.4f}):")
                print(f"   {query_scores[0][2][:80]}...")
                
                print(f"📉 需要改进的查询 (分数: {query_scores[-1][1]:.4f}):")
                print(f"   {query_scores[-1][2][:80]}...")
        
        print("\n✅ 结果分析完成")
        
    except Exception as e:
        print(f"❌ 结果分析失败: {str(e)}")


def main():
    """主演示函数"""
    print("🎉 多模态RAG评估系统演示")
    print("本演示将展示如何使用评估系统对多文档RAG进行全面评估")
    
    # 检查基本环境
    print("\n🔧 环境检查:")
    
    # 检查GPU
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU可用: {torch.cuda.get_device_name(9)}")
        else:
            print("⚠️  GPU不可用，将使用CPU（速度较慢）")
    except ImportError:
        print("❌ PyTorch未安装")
        return
    
    # 检查deepeval
    try:
        import deepeval
        print("✅ DeepEval已安装")
    except ImportError:
        print("❌ DeepEval未安装，请运行: pip install deepeval")
        return
    
    # 检查文档目录
    if os.path.exists('contents'):
        pdf_files = [f for f in os.listdir('contents') if f.endswith('.pdf')]
        print(f"✅ 找到 {len(pdf_files)} 个PDF文档")
    else:
        print("⚠️  contents目录不存在，请创建并放入PDF文档")
    
    print("\n" + "="*80)
    print("开始演示...")
    print("="*80)
    
    # 运行演示
    try:
        # 1. 单个查询评估
        demo_single_query_evaluation()
        
        # 2. 批量评估
        demo_batch_evaluation()
        
        # 3. 结果分析
        demo_result_analysis()
        
        print("\n" + "="*80)
        print("🎉 演示完成！")
        print("="*80)
        print("接下来您可以:")
        print("1. 修改queries列表添加更多测试查询")
        print("2. 调整评估阈值和模型参数")
        print("3. 分析evaluation_results/目录下的详细结果")
        print("4. 基于评估结果优化RAG系统")
        
    except KeyboardInterrupt:
        print("\n⏹️  演示被用户中断")
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {str(e)}")
        print("请检查环境配置和文档文件")


if __name__ == "__main__":
    main()
