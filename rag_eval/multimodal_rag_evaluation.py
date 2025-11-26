#!/usr/bin/env python3
"""
多模态RAG评估脚本
基于deepeval框架对Nvidia FAISS MultiDoc RAG系统进行全面评估

主要功能：
1. 多模态RAG指标评估（Answer Relevancy, Faithfulness, Contextual Precision/Recall/Relevancy）
2. 支持批量评估测试用例
3. 生成详细的评估报告
4. 与现有的Nvidia_FAISS_MultiDoc.ipynb集成

基于deepeval文档和Nvidia_FAISS_MultiDoc.ipynb构建
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# DeepEval imports
from deepeval import evaluate
from deepeval.metrics import (
    MultimodalAnswerRelevancyMetric,
    MultimodalContextualPrecisionMetric,
    MultimodalContextualRecallMetric,
    MultimodalContextualRelevancyMetric,
    MultimodalFaithfulnessMetric,
)
from deepeval.test_case import MLLMImage, MLLMTestCase

# 现有的RAG组件
from nvidia_rag_with_faiss import (
    NvidiaRAGPipeline, 
    GPUMemoryMonitor,
    ImageReranker,
    VQAModel
)
from pdf2image import convert_from_path
from transformers import AutoModel

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('multimodal_rag_evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MultimodalRAGEvaluator:
    """多模态RAG评估器"""
    
    def __init__(
        self,
        device: int = 9,
        model_name: str = "gpt-4o",
        threshold: float = 0.7,
        include_reason: bool = True,
        verbose_mode: bool = False
    ):
        """
        初始化评估器
        
        Args:
            device: GPU设备编号
            model_name: 用于评估的模型名称
            threshold: 评估阈值
            include_reason: 是否包含评估原因
            verbose_mode: 是否启用详细模式
        """
        self.device = device
        self.model_name = model_name
        self.threshold = threshold
        self.include_reason = include_reason
        self.verbose_mode = verbose_mode
        
        # 设置GPU设备
        torch.cuda.set_device(device)
        
        # 初始化内存监控
        self.memory_monitor = GPUMemoryMonitor(device)
        
        # 初始化评估指标
        self._init_metrics()
        
        # RAG组件（延迟加载）
        self.rag_pipeline = None
        self.reranker = None
        self.vqa_model = None
        self.docid2images = {}
        
        logger.info(f"MultimodalRAGEvaluator initialized with device={device}, model={model_name}")
    
    def _init_metrics(self):
        """初始化所有评估指标"""
        common_params = {
            "threshold": self.threshold,
            "model": self.model_name,
            "include_reason": self.include_reason,
            "verbose_mode": self.verbose_mode,
            "async_mode": True
        }
        
        self.metrics = {
            "answer_relevancy": MultimodalAnswerRelevancyMetric(**common_params),
            "faithfulness": MultimodalFaithfulnessMetric(**common_params),
            "contextual_precision": MultimodalContextualPrecisionMetric(**common_params),
            "contextual_recall": MultimodalContextualRecallMetric(**common_params),
            "contextual_relevancy": MultimodalContextualRelevancyMetric(**common_params)
        }
        
        logger.info(f"Initialized {len(self.metrics)} evaluation metrics")
    
    def load_rag_components(
        self,
        retriever_model_name: str = 'nvidia/llama-nemoretriever-colembed-3b-v1',
        reranker_model_name: str = "monovlm",
        vqa_model_name: str = "doubao-seed-1-6-vision-250815"
    ):
        """加载RAG组件"""
        logger.info("Loading RAG components...")
        
        # 加载Retriever模型
        self.memory_monitor.print_memory("Before loading retriever")
        
        retriever_model = AutoModel.from_pretrained(
            retriever_model_name,
            device_map=f'cuda:{self.device}',
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            revision='50c36f4d5271c6851aa08bd26d69f6e7ca8b870c',
        ).eval()
        
        self.memory_monitor.print_memory("After loading retriever")
        
        # 初始化RAG Pipeline
        self.rag_pipeline = NvidiaRAGPipeline(
            retriever_model=retriever_model,
            use_faiss=True,
            faiss_config={
                "embedding_dim": 3072,
                "index_type": "ivfflat",
                "nlist": 20,  # 默认值，会根据文档数量调整
                "use_gpu": False,  # 避免保存时的内存问题
            },
            device=self.device
        )
        
        # 加载Reranker
        self.memory_monitor.print_memory("Before loading reranker")
        
        self.reranker = ImageReranker(
            model_name=reranker_model_name,
            device=f"cuda:{self.device}",
            use_fast=True
        )
        
        self.memory_monitor.print_memory("After loading reranker")
        
        # 初始化VQA模型
        self.vqa_model = VQAModel(model_name=vqa_model_name)
        
        logger.info("✓ All RAG components loaded successfully")
    
    def load_documents(self, documents: Dict[str, str]) -> Dict[str, List]:
        """
        加载多个PDF文档
        
        Args:
            documents: 文档ID到PDF路径的映射
            
        Returns:
            文档ID到图片列表的映射
        """
        logger.info(f"Loading {len(documents)} documents...")
        
        docid2images = {}
        total_pages = 0
        
        for doc_id, pdf_path in documents.items():
            if not os.path.exists(pdf_path):
                logger.warning(f"PDF file not found: {pdf_path}")
                continue
                
            logger.info(f"Loading document: {doc_id}")
            images = convert_from_path(pdf_path, dpi=200)
            docid2images[doc_id] = images
            total_pages += len(images)
            logger.info(f"  ✓ {doc_id}: {len(images)} pages")
        
        self.docid2images = docid2images
        logger.info(f"✓ Loaded {len(docid2images)} documents with {total_pages} total pages")
        
        return docid2images
    
    def build_index(self, save_dir: Optional[str] = None):
        """构建多文档索引"""
        if not self.rag_pipeline:
            raise ValueError("RAG pipeline not loaded. Call load_rag_components() first.")
        
        if not self.docid2images:
            raise ValueError("No documents loaded. Call load_documents() first.")
        
        logger.info("Building multi-document index...")
        
        # 编码文档
        self.rag_pipeline.encode_documents(
            docid2images=self.docid2images,
            batch_size=8
        )
        
        # 构建索引
        total_pages = sum(len(images) for images in self.docid2images.values())
        
        # 调整nlist参数
        if self.rag_pipeline.faiss_manager:
            self.rag_pipeline.faiss_manager.nlist = int(np.sqrt(total_pages))
        
        self.rag_pipeline.build_unified_index(save_dir=save_dir)
        
        logger.info("✓ Multi-document index built successfully")
    
    def create_test_cases_from_queries(
        self,
        queries: List[str],
        expected_outputs: Optional[List[str]] = None,
        top_k: int = 10,
        rerank_top_k: int = 5
    ) -> List[MLLMTestCase]:
        """
        从查询创建测试用例
        
        Args:
            queries: 查询列表
            expected_outputs: 期望输出列表（可选）
            top_k: 检索的top-k结果
            rerank_top_k: 重排序后的top-k结果
            
        Returns:
            MLLM测试用例列表
        """
        if not self.rag_pipeline or not self.reranker or not self.vqa_model:
            raise ValueError("RAG components not loaded. Call load_rag_components() first.")
        
        logger.info(f"Creating test cases for {len(queries)} queries...")
        
        test_cases = []
        
        # 批量检索
        retrieval_results = self.rag_pipeline.retrieve_multi_doc(
            queries=queries,
            top_k=top_k,
            single_page_per_doc=False
        )
        
        # 准备候选图片
        all_candidate_images = []
        for query_results in retrieval_results:
            candidate_images = []
            for result in query_results:
                doc_id = result['doc_id']
                page_idx = result['page_idx']
                candidate_images.append(self.docid2images[doc_id][page_idx])
            all_candidate_images.append(candidate_images)
        
        # 批量重排序
        rerank_results = self.reranker.rerank_batch(
            queries=queries,
            all_images_list=all_candidate_images,
            top_k=rerank_top_k
        )
        
        # 准备VQA图片
        all_vqa_images = []
        for i, query_rerank_results in enumerate(rerank_results):
            vqa_images = [
                all_candidate_images[i][rr['doc_id']] 
                for rr in query_rerank_results[:rerank_top_k]
            ]
            all_vqa_images.append(vqa_images)
        
        # 批量VQA
        vqa_answers = self.vqa_model.answer_batch_with_multiple_images(
            queries=queries,
            all_images_list=all_vqa_images,
            max_tokens=1024
        )
        
        # 创建测试用例
        for i, query in enumerate(queries):
            # 获取检索上下文（重排序后的图片）
            retrieval_context = []
            for rr in rerank_results[i][:rerank_top_k]:
                img = all_candidate_images[i][rr['doc_id']]
                retrieval_context.append(MLLMImage(img))
            
            # 创建测试用例
            test_case = MLLMTestCase(
                input=[query],
                actual_output=[vqa_answers[i]],
                retrieval_context=retrieval_context,
                expected_output=[expected_outputs[i]] if expected_outputs and i < len(expected_outputs) else None
            )
            
            test_cases.append(test_case)
        
        logger.info(f"✓ Created {len(test_cases)} test cases")
        return test_cases
    
    def evaluate_test_cases(
        self,
        test_cases: List[MLLMTestCase],
        metrics_to_use: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        评估测试用例
        
        Args:
            test_cases: 测试用例列表
            metrics_to_use: 要使用的指标名称列表（None表示使用所有指标）
            
        Returns:
            评估结果
        """
        if metrics_to_use is None:
            metrics_to_use = list(self.metrics.keys())
        
        # 过滤指标
        selected_metrics = []
        for metric_name in metrics_to_use:
            if metric_name in self.metrics:
                selected_metrics.append(self.metrics[metric_name])
            else:
                logger.warning(f"Unknown metric: {metric_name}")
        
        if not selected_metrics:
            raise ValueError("No valid metrics selected")
        
        logger.info(f"Evaluating {len(test_cases)} test cases with {len(selected_metrics)} metrics...")
        
        # 过滤需要expected_output的指标
        metrics_needing_expected = ["contextual_precision", "contextual_recall"]
        filtered_metrics = []
        
        for i, metric in enumerate(selected_metrics):
            metric_name = metrics_to_use[i]
            
            if metric_name in metrics_needing_expected:
                # 检查是否所有测试用例都有expected_output
                has_expected = all(tc.expected_output is not None for tc in test_cases)
                if not has_expected:
                    logger.warning(f"Skipping {metric_name} - requires expected_output for all test cases")
                    continue
            
            filtered_metrics.append(metric)
        
        if not filtered_metrics:
            raise ValueError("No metrics can be evaluated with the given test cases")
        
        # 执行评估
        try:
            evaluation_results = evaluate(test_cases, filtered_metrics)
            logger.info("✓ Evaluation completed successfully")
            return evaluation_results
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            raise
    
    def run_comprehensive_evaluation(
        self,
        documents: Dict[str, str],
        queries: List[str],
        expected_outputs: Optional[List[str]] = None,
        save_index: bool = True,
        index_save_dir: str = "./faiss_index/evaluation",
        report_save_dir: str = "./evaluation_reports"
    ) -> Dict[str, Any]:
        """
        运行全面评估
        
        Args:
            documents: 文档映射
            queries: 查询列表
            expected_outputs: 期望输出列表
            save_index: 是否保存索引
            index_save_dir: 索引保存目录
            report_save_dir: 报告保存目录
            
        Returns:
            完整的评估结果
        """
        logger.info("Starting comprehensive evaluation...")
        
        # 创建保存目录
        os.makedirs(report_save_dir, exist_ok=True)
        if save_index:
            os.makedirs(index_save_dir, exist_ok=True)
        
        # 1. 加载组件
        if not self.rag_pipeline:
            self.load_rag_components()
        
        # 2. 加载文档
        self.load_documents(documents)
        
        # 3. 构建索引
        self.build_index(save_dir=index_save_dir if save_index else None)
        
        # 4. 创建测试用例
        test_cases = self.create_test_cases_from_queries(
            queries=queries,
            expected_outputs=expected_outputs
        )
        
        # 5. 执行评估
        evaluation_results = self.evaluate_test_cases(test_cases)
        
        # 6. 生成报告
        report = self._generate_evaluation_report(
            evaluation_results,
            queries,
            documents,
            test_cases
        )
        
        # 7. 保存报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(report_save_dir, f"evaluation_report_{timestamp}.json")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"✓ Comprehensive evaluation completed. Report saved to: {report_file}")
        
        return report
    
    def _generate_evaluation_report(
        self,
        evaluation_results: Dict[str, Any],
        queries: List[str],
        documents: Dict[str, str],
        test_cases: List[MLLMTestCase]
    ) -> Dict[str, Any]:
        """生成评估报告"""
        
        report = {
            "evaluation_summary": {
                "timestamp": datetime.now().isoformat(),
                "model_name": self.model_name,
                "threshold": self.threshold,
                "num_queries": len(queries),
                "num_documents": len(documents),
                "total_pages": sum(len(images) for images in self.docid2images.values())
            },
            "documents": documents,
            "queries": queries,
            "test_results": [],
            "metrics_summary": {},
            "overall_performance": {}
        }
        
        # 处理每个测试用例的结果
        for i, test_case in enumerate(test_cases):
            test_result = {
                "query_index": i,
                "query": queries[i],
                "actual_output": test_case.actual_output[0] if test_case.actual_output else None,
                "expected_output": test_case.expected_output[0] if test_case.expected_output else None,
                "retrieval_context_count": len(test_case.retrieval_context) if test_case.retrieval_context else 0,
                "metrics_scores": {}
            }
            
            # 提取各指标的分数和原因
            for metric_name, metric in self.metrics.items():
                if hasattr(metric, 'score') and metric.score is not None:
                    test_result["metrics_scores"][metric_name] = {
                        "score": float(metric.score),
                        "reason": getattr(metric, 'reason', None) if self.include_reason else None,
                        "passed": float(metric.score) >= self.threshold
                    }
            
            report["test_results"].append(test_result)
        
        # 计算指标汇总
        for metric_name in self.metrics.keys():
            scores = []
            passed_count = 0
            
            for test_result in report["test_results"]:
                if metric_name in test_result["metrics_scores"]:
                    score = test_result["metrics_scores"][metric_name]["score"]
                    scores.append(score)
                    if test_result["metrics_scores"][metric_name]["passed"]:
                        passed_count += 1
            
            if scores:
                report["metrics_summary"][metric_name] = {
                    "average_score": np.mean(scores),
                    "min_score": np.min(scores),
                    "max_score": np.max(scores),
                    "std_score": np.std(scores),
                    "pass_rate": passed_count / len(scores),
                    "total_tests": len(scores)
                }
        
        # 计算整体性能
        all_scores = []
        all_passed = 0
        total_tests = 0
        
        for metric_summary in report["metrics_summary"].values():
            all_scores.extend([metric_summary["average_score"]])
            all_passed += metric_summary["pass_rate"] * metric_summary["total_tests"]
            total_tests += metric_summary["total_tests"]
        
        if all_scores:
            report["overall_performance"] = {
                "average_score_across_metrics": np.mean(all_scores),
                "overall_pass_rate": all_passed / total_tests if total_tests > 0 else 0,
                "total_metric_evaluations": total_tests
            }
        
        return report


def main():
    """主函数 - 示例用法"""
    
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
        "各公司在环境保护方面的主要措施是什么？",
        "社会责任报告中提到的员工培训情况如何？"
    ]
    
    # 期望输出（可选，用于contextual precision和recall评估）
    expected_outputs = [
        "各公司在GRI403-3职业健康服务方面的披露形式和深度存在显著差异...",
        "各公司采取了多样化的环境保护措施，包括节能减排、绿色技术应用等...",
        "各公司都重视员工培训，但在培训内容、覆盖面和效果评估方面有所不同..."
    ]
    
    # 初始化评估器
    evaluator = MultimodalRAGEvaluator(
        device=9,
        model_name="gpt-4o",
        threshold=0.7,
        include_reason=True,
        verbose_mode=False
    )
    
    try:
        # 运行全面评估
        results = evaluator.run_comprehensive_evaluation(
            documents=documents,
            queries=queries,
            expected_outputs=expected_outputs,
            save_index=True,
            index_save_dir="./faiss_index/evaluation",
            report_save_dir="./evaluation_reports"
        )
        
        # 打印简要结果
        print("\n" + "="*80)
        print("评估结果摘要")
        print("="*80)
        
        if "overall_performance" in results:
            overall = results["overall_performance"]
            print(f"整体平均分数: {overall.get('average_score_across_metrics', 0):.4f}")
            print(f"整体通过率: {overall.get('overall_pass_rate', 0):.2%}")
        
        if "metrics_summary" in results:
            print("\n各指标表现:")
            for metric_name, summary in results["metrics_summary"].items():
                print(f"  {metric_name}:")
                print(f"    平均分数: {summary['average_score']:.4f}")
                print(f"    通过率: {summary['pass_rate']:.2%}")
        
        print("\n详细报告已保存到: evaluation_reports/")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
