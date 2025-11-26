#!/usr/bin/env python3
"""
简化的多模态RAG评估脚本
基于deepeval框架和Nvidia_FAISS_MultiDoc.ipynb

主要功能：
1. 集成现有的RAG pipeline
2. 使用deepeval的多模态指标进行评估
3. 生成评估报告

使用方法：
python simple_multimodal_evaluation.py
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from deepeval import evaluate
from deepeval.metrics import (
    MultimodalAnswerRelevancyMetric,
    MultimodalContextualRelevancyMetric,
    MultimodalFaithfulnessMetric,
)
from deepeval.test_case import MLLMImage, MLLMTestCase
from nvidia_rag_with_faiss import GPUMemoryMonitor, ImageReranker, NvidiaRAGPipeline, VQAModel
from pdf2image import convert_from_path
from transformers import AutoModel

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleMultimodalEvaluator:
    """简化的多模态RAG评估器"""
    
    def __init__(self, device: int = 9):
        self.device = device
        torch.cuda.set_device(device)
        self.memory_monitor = GPUMemoryMonitor(device)
        
        # 初始化组件（延迟加载）
        self.rag_pipeline = None
        self.reranker = None
        self.vqa_model = None
        self.docid2images = {}
        
        logger.info(f"SimpleMultimodalEvaluator initialized on device {device}")
    
    def setup_rag_system(self):
        """设置RAG系统组件"""
        logger.info("Setting up RAG system...")
        
        # 1. 加载Retriever模型
        self.memory_monitor.print_memory("Before loading retriever")
        
        retriever_model = AutoModel.from_pretrained(
            'nvidia/llama-nemoretriever-colembed-3b-v1',
            device_map=f'cuda:{self.device}',
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            revision='50c36f4d5271c6851aa08bd26d69f6e7ca8b870c',
        ).eval()
        
        self.memory_monitor.print_memory("After loading retriever")
        
        # 2. 初始化RAG Pipeline
        self.rag_pipeline = NvidiaRAGPipeline(
            retriever_model=retriever_model,
            use_faiss=True,
            faiss_config={
                "embedding_dim": 3072,
                "index_type": "ivfflat",
                "nlist": 20,
                "use_gpu": False,
            },
            device=self.device
        )
        
        # 3. 加载Reranker
        self.memory_monitor.print_memory("Before loading reranker")
        
        self.reranker = ImageReranker(
            model_name="monovlm",
            device=f"cuda:{self.device}",
            use_fast=True
        )
        
        self.memory_monitor.print_memory("After loading reranker")
        
        # 4. 初始化VQA模型
        self.vqa_model = VQAModel(model_name="doubao-seed-1-6-vision-250815")
        
        logger.info("✓ RAG system setup completed")
    
    def load_documents(self, documents: Dict[str, str]):
        """加载文档"""
        logger.info(f"Loading {len(documents)} documents...")
        
        self.docid2images = {}
        total_pages = 0
        
        for doc_id, pdf_path in documents.items():
            if not os.path.exists(pdf_path):
                logger.warning(f"PDF file not found: {pdf_path}")
                continue
                
            logger.info(f"Loading document: {doc_id}")
            images = convert_from_path(pdf_path, dpi=200)
            self.docid2images[doc_id] = images
            total_pages += len(images)
            logger.info(f"  ✓ {doc_id}: {len(images)} pages")
        
        logger.info(f"✓ Loaded {len(self.docid2images)} documents with {total_pages} total pages")
        
        # 构建索引
        logger.info("Building index...")
        self.rag_pipeline.encode_documents(
            docid2images=self.docid2images,
            batch_size=8
        )
        
        # 调整nlist参数并构建索引
        if self.rag_pipeline.faiss_manager:
            self.rag_pipeline.faiss_manager.nlist = int(np.sqrt(total_pages))
        self.rag_pipeline.build_unified_index(save_dir=None)  # 不保存到磁盘
        
        logger.info("✓ Index built successfully")
    
    def create_test_case(self, query: str, top_k: int = 10, rerank_k: int = 5) -> MLLMTestCase:
        """为单个查询创建测试用例"""
        
        # 1. 检索
        retrieval_results = self.rag_pipeline.retrieve_multi_doc(
            queries=[query],
            top_k=top_k,
            single_page_per_doc=False
        )
        
        # 2. 准备候选图片
        candidate_images = []
        for result in retrieval_results[0]:
            doc_id = result['doc_id']
            page_idx = result['page_idx']
            candidate_images.append(self.docid2images[doc_id][page_idx])
        
        # 3. 重排序
        rerank_results = self.reranker.rerank_batch(
            queries=[query],
            all_images_list=[candidate_images],
            top_k=rerank_k
        )
        
        # 4. 准备VQA图片
        vqa_images = [
            candidate_images[rr['doc_id']] 
            for rr in rerank_results[0][:rerank_k]
        ]
        
        # 5. 生成答案
        vqa_answer = self.vqa_model.answer_batch_with_multiple_images(
            queries=[query],
            all_images_list=[vqa_images],
            max_tokens=1024
        )[0]
        
        # 6. 创建检索上下文
        retrieval_context = [MLLMImage(img) for img in vqa_images]
        
        # 7. 创建测试用例
        test_case = MLLMTestCase(
            input=[query],
            actual_output=[vqa_answer],
            retrieval_context=retrieval_context
        )
        
        return test_case
    
    def evaluate_query(self, query: str, model_name: str = "gpt-4o") -> Dict[str, Any]:
        """评估单个查询"""
        logger.info(f"Evaluating query: {query}")
        
        # 创建测试用例
        test_case = self.create_test_case(query)
        
        # 定义评估指标
        metrics = [
            MultimodalAnswerRelevancyMetric(
                model=model_name,
                threshold=0.7,
                include_reason=True
            ),
            MultimodalFaithfulnessMetric(
                model=model_name,
                threshold=0.7,
                include_reason=True
            ),
            MultimodalContextualRelevancyMetric(
                model=model_name,
                threshold=0.7,
                include_reason=True
            )
        ]
        
        # 执行评估
        try:
            logger.info("Running evaluation...")
            evaluate([test_case], metrics)
            
            # 收集结果
            results = {
                "query": query,
                "actual_output": test_case.actual_output[0],
                "retrieval_context_count": len(test_case.retrieval_context),
                "metrics": {}
            }
            
            for i, metric in enumerate(metrics):
                metric_name = [
                    "answer_relevancy",
                    "faithfulness", 
                    "contextual_relevancy"
                ][i]
                
                results["metrics"][metric_name] = {
                    "score": float(metric.score) if hasattr(metric, 'score') and metric.score is not None else None,
                    "reason": getattr(metric, 'reason', None),
                    "passed": float(metric.score) >= 0.7 if hasattr(metric, 'score') and metric.score is not None else False
                }
            
            logger.info("✓ Evaluation completed")
            return results
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            return {
                "query": query,
                "error": str(e),
                "metrics": {}
            }
    
    def run_batch_evaluation(
        self, 
        documents: Dict[str, str], 
        queries: List[str],
        model_name: str = "gpt-4o"
    ) -> Dict[str, Any]:
        """运行批量评估"""
        logger.info("Starting batch evaluation...")
        
        # 设置系统
        if not self.rag_pipeline:
            self.setup_rag_system()
        
        # 加载文档
        self.load_documents(documents)
        
        # 评估每个查询
        results = {
            "evaluation_info": {
                "timestamp": datetime.now().isoformat(),
                "model_name": model_name,
                "num_documents": len(documents),
                "num_queries": len(queries),
                "documents": documents
            },
            "query_results": [],
            "summary": {}
        }
        
        for query in queries:
            query_result = self.evaluate_query(query, model_name)
            results["query_results"].append(query_result)
        
        # 计算汇总统计
        metric_names = ["answer_relevancy", "faithfulness", "contextual_relevancy"]
        summary = {}
        
        for metric_name in metric_names:
            scores = []
            passed_count = 0
            
            for query_result in results["query_results"]:
                if metric_name in query_result.get("metrics", {}):
                    metric_data = query_result["metrics"][metric_name]
                    if metric_data["score"] is not None:
                        scores.append(metric_data["score"])
                        if metric_data["passed"]:
                            passed_count += 1
            
            if scores:
                summary[metric_name] = {
                    "average_score": np.mean(scores),
                    "min_score": np.min(scores),
                    "max_score": np.max(scores),
                    "pass_rate": passed_count / len(scores)
                }
        
        results["summary"] = summary
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("evaluation_results", exist_ok=True)
        result_file = f"evaluation_results/evaluation_{timestamp}.json"
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"✓ Batch evaluation completed. Results saved to: {result_file}")
        return results


def main():
    """主函数"""
    
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
    
    # 初始化评估器
    evaluator = SimpleMultimodalEvaluator(device=9)
    
    try:
        # 运行评估
        results = evaluator.run_batch_evaluation(
            documents=documents,
            queries=queries,
            model_name="gpt-4o"
        )
        
        # 打印结果摘要
        print("\n" + "="*80)
        print("评估结果摘要")
        print("="*80)
        
        if "summary" in results:
            for metric_name, summary in results["summary"].items():
                print(f"\n{metric_name}:")
                print(f"  平均分数: {summary['average_score']:.4f}")
                print(f"  分数范围: {summary['min_score']:.4f} - {summary['max_score']:.4f}")
                print(f"  通过率: {summary['pass_rate']:.2%}")
        
        print(f"\n详细结果已保存到: evaluation_results/")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
