"""
Nvidia NemoRetriever RAG Pipeline with FAISS Indexing
重构版本：添加向量索引支持，提升大规模文档检索性能
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import gc
import json
import torch
import numpy as np
import faiss
from typing import List, Dict, Tuple, Optional
from PIL import Image
from pathlib import Path
import pickle
from tqdm.auto import tqdm
from openai import OpenAI
from colpali_engine.models import ColQwen2, ColQwen2Processor
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from langchain.text_splitter import RecursiveCharacterTextSplitter
from models.gme import GmeQwen2VL
import utils.crop_papers
# from MMDocIRContentJudger.qwen25vl_judger import ContentJudger
# import easyocr

class GPUMemoryMonitor:
    """GPU显存监控工具"""
    
    def __init__(self, device: int = 0):
        self.device = device
    
    def print_memory(self, stage: str = ""):
        """打印当前GPU显存使用情况"""
        if torch.cuda.is_available():
            torch.cuda.set_device(self.device)
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"[{stage}] GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    
    @staticmethod
    def clear_memory():
        """清理GPU显存"""
        gc.collect()
        torch.cuda.empty_cache()


class FAISSIndexManager:
    """FAISS索引管理器"""
    
    def __init__(
        self,
        embedding_dim: int = 3072,  # Nvidia NemoRetriever 的实际维度
        index_type: str = "flat",  # 'flat', 'ivfflat', 'ivfpq'
        nlist: int = 100,  # IVF聚类中心数量
        use_gpu: bool = False,
        gpu_id: int = 0
    ):
        """
        初始化FAISS索引
        
        Args:
            embedding_dim: 嵌入向量维度
            index_type: 索引类型
                - 'flat': 精确搜索，适合小规模(<10K)
                - 'ivfflat': 倒排索引，适合中等规模(10K-1M)
                - 'ivfpq': 乘积量化，适合大规模(>1M)
            nlist: IVF索引的聚类中心数量
            use_gpu: 是否使用GPU加速索引
            gpu_id: GPU设备ID
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.nlist = nlist
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        
        self.index = None
        self.page_id_map = []  # 存储 page_uid 映射
        self.is_trained = False
        
    def _create_index(self) -> faiss.Index:
        """创建FAISS索引"""
        if self.index_type == "flat":
            # 精确搜索索引
            index = faiss.IndexFlatIP(self.embedding_dim)  # 内积相似度
            
        elif self.index_type == "ivfflat":
            # IVF倒排索引
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, self.nlist)
            
        elif self.index_type == "ivfpq":
            # IVF + 乘积量化
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            m = 8  # 子向量数量
            nbits = 8  # 每个子向量的比特数
            index = faiss.IndexIVFPQ(quantizer, self.embedding_dim, self.nlist, m, nbits)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        # GPU加速
        if self.use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, self.gpu_id, index)
            print(f"✓ FAISS索引已移至GPU {self.gpu_id}")
        
        return index
    
    def build_index(
        self,
        embeddings: np.ndarray,
        page_ids: List[str],
        batch_size: int = 8
    ):
        """
        构建FAISS索引
        
        Args:
            embeddings: 嵌入向量 (n_pages, n_tokens, dim)
            page_ids: 页面ID列表 (n_pages,)
            batch_size: 批处理大小
        """
        print(f"\n{'='*80}")
        print(f"构建FAISS索引 (类型: {self.index_type})")
        print(f"{'='*80}")
        
        # 展平嵌入向量: (n_pages, n_tokens, dim) -> (n_total_tokens, dim)
        all_token_embeddings = []
        token_to_page_map = []
        
        print("展平嵌入向量...")
        for page_idx, page_emb in enumerate(tqdm(embeddings, desc="Processing pages")):
            # page_emb: (n_tokens, dim)
            n_tokens = page_emb.shape[0]
            all_token_embeddings.append(page_emb)
            token_to_page_map.extend([page_ids[page_idx]] * n_tokens)
        
        # 合并所有token嵌入
        all_token_embeddings = np.vstack(all_token_embeddings).astype(np.float32)
        print(f"✓ 总共 {len(all_token_embeddings)} 个token嵌入")
        
        # 归一化（用于内积相似度）
        faiss.normalize_L2(all_token_embeddings)
        
        # 创建索引
        self.index = self._create_index()
        self.page_id_map = token_to_page_map
        
        # 训练索引（IVF类型需要）
        if self.index_type in ["ivfflat", "ivfpq"]:
            print(f"训练索引 (nlist={self.nlist})...")
            # 使用部分数据训练（最多100K样本）
            train_size = min(100000, len(all_token_embeddings))
            train_data = all_token_embeddings[:train_size]
            self.index.train(train_data)
            self.is_trained = True
            print("✓ 索引训练完成")
        
        # 添加向量到索引
        print("添加向量到索引...")
        n_batches = (len(all_token_embeddings) + batch_size - 1) // batch_size
        
        for i in tqdm(range(n_batches), desc="Adding vectors"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(all_token_embeddings))
            batch = all_token_embeddings[start_idx:end_idx]
            self.index.add(batch)
        
        print(f"✓ 索引构建完成！总共 {self.index.ntotal} 个向量")
        print(f"{'='*80}\n")
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        nprobe: int = 10
    ) -> List[Tuple[str, float]]:
        """
        使用MaxSim策略搜索最相关的页面
        
        Args:
            query_embedding: 查询嵌入 (n_query_tokens, dim)
            top_k: 返回top-k个页面
            nprobe: IVF索引的探测聚类数量
            
        Returns:
            [(page_id, score), ...] 按分数降序排列
        """
        if self.index is None:
            raise ValueError("索引未构建，请先调用 build_index()")
        
        # 设置nprobe（仅对IVF索引有效）
        if self.index_type in ["ivfflat", "ivfpq"]:
            if hasattr(self.index, 'nprobe'):
                self.index.nprobe = nprobe
        
        # 归一化查询向量
        query_embedding = query_embedding.astype(np.float32)
        faiss.normalize_L2(query_embedding)
        print(f"query_embedding.shape: {query_embedding.shape}")
        # 对每个查询token搜索最近邻
        k_per_token = top_k * 10  # 每个token搜索更多候选
        D, I = self.index.search(query_embedding, k_per_token)
        
        # MaxSim聚合：对每个页面，取所有查询token的最大相似度之和
        page_scores = {}
        
        for q_idx in range(len(query_embedding)):
            # 当前查询token的最近邻
            token_scores = {}
            
            for nn_idx in range(k_per_token):
                token_idx = I[q_idx, nn_idx]
                if token_idx < 0 or token_idx >= len(self.page_id_map):
                    continue
                
                page_id = self.page_id_map[token_idx]
                score = D[q_idx, nn_idx]
                
                # MaxSim: 每个页面保留最大分数
                if page_id not in token_scores:
                    token_scores[page_id] = score
                else:
                    token_scores[page_id] = max(token_scores[page_id], score)
            
            # 累加所有查询token的分数
            for page_id, score in token_scores.items():
                if page_id in page_scores:
                    page_scores[page_id] += score
                else:
                    page_scores[page_id] = score
        
        # 排序并返回top-k
        sorted_pages = sorted(page_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_pages[:top_k]
    
    def save(self, save_dir: str):
        """保存索引到磁盘"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存FAISS索引
        if self.use_gpu:
            # GPU索引需要先转回CPU
            # 对于大规模索引，直接转换可能导致内存溢出
            # 解决方案：重新创建CPU索引并重新训练/添加
            print("⚠️ GPU索引较大，正在转换为CPU索引...")
            try:
                cpu_index = faiss.index_gpu_to_cpu(self.index)
                faiss.write_index(cpu_index, str(save_dir / "faiss.index"))
            except RuntimeError as e:
                print(f"⚠️ GPU->CPU转换失败: {e}")
                print("建议：使用 use_gpu=False 创建CPU索引，或不保存索引")
                # 只保存配置和映射，不保存索引
                print("跳过索引保存，仅保存配置和映射")
        else:
            faiss.write_index(self.index, str(save_dir / "faiss.index"))
        
        # 保存page_id映射
        with open(save_dir / "page_id_map.pkl", "wb") as f:
            pickle.dump(self.page_id_map, f)
        
        # 保存配置
        config = {
            "embedding_dim": self.embedding_dim,
            "index_type": self.index_type,
            "nlist": self.nlist,
        }
        with open(save_dir / "config.pkl", "wb") as f:
            pickle.dump(config, f)
        
        print(f"✓ 配置和映射已保存到 {save_dir}")
    
    def load(self, load_dir: str):
        """从磁盘加载索引"""
        load_dir = Path(load_dir)
        
        # 加载配置
        with open(load_dir / "config.pkl", "rb") as f:
            config = pickle.load(f)
        
        self.embedding_dim = config["embedding_dim"]
        self.index_type = config["index_type"]
        self.nlist = config["nlist"]
        
        # 加载FAISS索引
        self.index = faiss.read_index(str(load_dir / "faiss.index"))
        
        # GPU加速
        if self.use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, self.gpu_id, self.index)
        
        # 加载page_id映射
        with open(load_dir / "page_id_map.pkl", "rb") as f:
            self.page_id_map = pickle.load(f)
        
        print(f"✓ 索引已从 {load_dir} 加载")
        print(f"  - 索引类型: {self.index_type}")
        print(f"  - 向量数量: {self.index.ntotal}")
        print(f"  - 页面数量: {len(set(self.page_id_map))}")




class ImageReranker:
    """图片重排序器（使用 MonoVLM）"""
    
    def __init__(self, model_name: str = "monovlm", device: str = "cuda:0", use_fast: bool = True):
        """
        初始化 Reranker
        
        Args:
            model_name: 模型名称，默认 "monovlm"
            device: 设备，例如 "cuda:0"
            use_fast: 是否使用快速处理器
        """
        try:
            from rerankers import Reranker
        except ImportError:
            raise ImportError("请安装 rerankers 库: pip install rerankers")
        
        self.device = device
        print(f"Loading {model_name} reranker on {device}...")
        self.ranker = Reranker(model_name, use_fast=use_fast, device=device)
        print("✓ Reranker loaded successfully")
    
    def rerank(
        self,
        query: str,
        images: List[Image.Image],
        top_k: int = 5
    ) -> List[Dict]:
        """
        对图片进行重排序
        
        Args:
            query: 查询文本
            images: PIL 图片列表
            top_k: 返回前 k 个结果
            
        Returns:
            重排序后的结果列表，每个元素包含 {rank, doc_id, score}
        """
        # Base64转换函数
        import base64
        from io import BytesIO
        # 转换图片为 base64
        base64_images = []
        for img in images:
            buffer = BytesIO()
            img.save(buffer, format="JPEG")
            buffer.seek(0)
            
            img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            base64_images.append(img_base64)
        
        # 执行重排序
        rerank_results = self.ranker.rank(query, base64_images)
        
        # 提取 top-k 结果
        results = []
        for i, doc in enumerate(rerank_results.top_k(top_k)):
            results.append({
                'rank': i + 1,
                'doc_id': doc.doc_id,
                'score': doc.score
            })
        
        return results
    
    def rerank_batch(
        self,
        queries: List[str],
        all_images_list: List[List[Image.Image]],
        top_k: int = 5
    ) -> List[List[Dict]]:
        """
        批量重排序
        
        Args:
            queries: 查询列表
            all_images_list: 每个查询对应的图片列表
            top_k: 每个查询返回前 k 个结果
            
        Returns:
            每个查询的重排序结果
        """
        all_rerank_results = []
        
        for query, images in zip(queries, all_images_list):
            rerank_results = self.rerank(query, images, top_k=top_k)
            all_rerank_results.append(rerank_results)
        
        return all_rerank_results

class ColQwenPDFRetriever:
    """ColQwen2-7B 文本到图像检索（单 PDF 文档）"""

    def __init__(self, colqwen_path: Optional[str] = None):
        if colqwen_path is None:
            colqwen_path = os.environ.get("COLQWEN2_7B_PATH")
        if not colqwen_path:
            raise ValueError(
                "ColQwen2-7B 模型路径未设置，请在初始化时传入 colqwen_path，"
                "或设置环境变量 COLQWEN2_7B_PATH"
            )
        self.model = ColQwen2.from_pretrained(
            colqwen_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
        ).eval()
        self.processor = ColQwen2Processor.from_pretrained(colqwen_path)

    @staticmethod
    def _resize_image(image: Image.Image) -> Image.Image:
        new_h, new_w = smart_resize(image.height, image.width)
        return image.resize((new_w, new_h))

    def search(self, query: str, images: List[Image.Image], top_k: int = 20) -> List[Dict]:
        """对单个查询在整本 PDF 上进行 ColQwen 检索"""
        if len(images) == 0:
            return []

        # 编码查询
        query_inputs = self.processor.process_queries([query]).to(self.model.device)
        with torch.no_grad():
            query_embedding = self.model(**query_inputs)

        # 编码所有页面图像
        image_embedding_list = []
        batch_size = 2
        for i in range(0, len(images), batch_size):
            batch_images = [self._resize_image(img) for img in images[i:i + batch_size]]
            image_inputs = self.processor.process_images(batch_images).to(self.model.device)
            with torch.no_grad():
                image_embeddings = self.model(**image_inputs)
            for emb in image_embeddings:
                image_embedding_list.append(emb.squeeze(0))

        # 计算相似度
        scores = self.processor.score_multi_vector(query_embedding, image_embedding_list)
        scores = scores[0].cpu().numpy()

        top_indices = np.argsort(scores).tolist()[::-1][:top_k]
        results: List[Dict] = []
        for rank, idx in enumerate(top_indices):
            results.append(
                {
                    "page_idx": int(idx),
                    "page_num": int(idx) + 1,
                    "score": float(scores[idx]),
                    "rank": rank + 1
                }
            )
        return results

    def search_batch(self, queries: List[str], images: List[Image.Image], top_k: int = 20) -> List[List[Dict]]:
        if isinstance(queries, str):
            queries = [queries]
        return [self.search(q, images, top_k=top_k) for q in queries]


def _gme_get_instruction(query: str) -> str:
    """从 mmdocir_2_gme_layout_retrieval 复用的指令模板"""
    default = "image"
    q = query.lower()
    if "figure" in q:
        default = "figure"
    if "table" in q:
        default = "table"
    if "page" in q:
        default = "page"
    template = f"Find an {default} that can solve the given question."
    return template


def _gme_build_page_instruct(page: int, num_page: int) -> str:
    """从 mmdocir_2_gme_layout_* 复用的页面描述模板"""
    template = f"This is the image on page {page + 1} of the {num_page} pages document. Describe the content in the page."
    return template


class GMELayoutPDFRetriever:
    """GME layout-level 文本到图像检索（单 PDF 文档）"""

    def __init__(self, gme_model: GmeQwen2VL):
        self.gme = gme_model
        self.page_embeddings: Optional[torch.Tensor] = None  # [n_pages, dim]
        self.num_pages: int = 0

    def build_index(self, images: List[Image.Image]):
        """预计算每一页的布局级嵌入"""
        if len(images) == 0:
            self.page_embeddings = None
            self.num_pages = 0
            return

        embeddings: List[torch.Tensor] = []
        num_pages = len(images)

        with torch.no_grad():
            for page_idx, img in enumerate(images):
                # 页面 + 布局切块
                np_img = np.array(img)
                layout_crops = utils.crop_papers.get_paper_layout(np_img)
                page_images = [img]
                page_images.extend(layout_crops)

                instruct = _gme_build_page_instruct(page_idx, num_pages)
                texts = [instruct] * len(page_images)

                page_embs = self.gme.get_fused_embeddings(texts=texts, images=page_images).cpu()
                # 对当前页的多个布局片段做 max pooling 得到单一向量
                page_vec, _ = torch.max(page_embs, dim=0)
                embeddings.append(page_vec)

        self.page_embeddings = torch.stack(embeddings, dim=0)  # [n_pages, dim]
        self.num_pages = num_pages

    def search(self, query: str, top_k: int = 20) -> List[Dict]:
        """在预计算的布局级索引上进行检索"""
        if self.page_embeddings is None or self.num_pages == 0:
            return []

        with torch.no_grad():
            query_emb = self.gme.get_text_embeddings(
                texts=[query],
                instruction=_gme_get_instruction(query),
            ).cpu()[0]  # [dim]

        # 计算与每一页的相似度
        scores = (self.page_embeddings * query_emb).sum(-1).numpy()
        indices = np.argsort(scores)[::-1][:top_k]

        results: List[Dict] = []
        for rank, idx in enumerate(indices):
            results.append(
                {
                    "page_idx": int(idx),
                    "page_num": int(idx) + 1,
                    "score": float(scores[idx]),
                    "rank": rank + 1
                }
            )
        return results


class DotsOCRPageTextExtractor:
    """从 dots_ocr 解析中提取每页纯文本的辅助类。

    预期输入目录结构示例：
        <ocr_output_dir>/xxx_page_0.jpg
        <ocr_output_dir>/xxx_page_0.md  # JSON list: [{"bbox": ..., "category": ..., "text": ...}, ...]

    作用：
        - 从每个 xxx_page_*.md 中抽取所有 item["text"], 拼接为单页文本
        - 按页号排序返回 List[str]（page0_text, page1_text, ...）
        - 可选在同目录下写入 xxx_page_0_text.md 等纯文本文件
    """

    def __init__(self, text_suffix: str = "_text") -> None:
        self.text_suffix = text_suffix

    def extract_page_texts(
        self,
        ocr_output_dir: str,
        save_plaintext_md: bool = True,
    ) -> List[str]:
        base_dir = Path(ocr_output_dir)
        if not base_dir.exists():
            raise FileNotFoundError(f"OCR output directory not found: {ocr_output_dir}")

        page_texts_map: Dict[int, str] = {}

        for md_file in sorted(base_dir.glob("*_page_*.md")):
            stem = md_file.stem  # e.g. xxx_page_0
            # 跳过已经是纯文本的文件（例如 *_text.md）
            if stem.endswith(self.text_suffix):
                continue

            try:
                _, page_part = stem.rsplit("_page_", 1)
                page_idx = int(page_part)
            except ValueError:
                # 文件名不符合预期模式时跳过
                continue

            raw = md_file.read_text(encoding="utf-8")
            try:
                data = json.loads(raw)
            except Exception:
                # 不是合法 JSON 时跳过该页
                continue

            if not isinstance(data, list):
                continue

            parts: List[str] = []
            for item in data:
                if isinstance(item, dict):
                    value = item.get("text")
                    if isinstance(value, str) and value.strip():
                        parts.append(value.strip())

            page_text = "\n".join(parts)

            if save_plaintext_md:
                text_md_path = md_file.with_name(stem + f"{self.text_suffix}.md")
                text_md_path.write_text(page_text, encoding="utf-8")

            page_texts_map[page_idx] = page_text

        # 按页号排序，构造 page_texts 列表
        page_texts: List[str] = []
        for idx in sorted(page_texts_map.keys()):
            page_texts.append(page_texts_map[idx])

        return page_texts


class GMETextPDFRetriever:
    """GME 文本到文本检索（单 PDF 文档）"""

    def __init__(
        self,
        gme_model: GmeQwen2VL,
        use_ocr: bool = True,
        ocr_langs: Optional[List[str]] = None,
    ):
        self.gme = gme_model
        self.use_ocr = use_ocr
        self.ocr_langs = ocr_langs or ["ch_sim", "en"]
        self.page_texts: Optional[List[str]] = None
        self.chunk_embeddings: Optional[torch.Tensor] = None  # [n_chunks, dim]
        self.chunk2page: Optional[List[int]] = None
        # OCR 抽取与文本分块+embedding 解耦：
        # 1) OCR / dots_ocr 生成 xxx_page_*.md（或使用 DotsOCRPageTextExtractor 得到 page_texts）
        # 2) 再基于 page_texts 进行分块和 GME 文本嵌入索引构建

    def build_index_from_texts(
        self,
        page_texts: List[str],
        chunk_size: int = 1000,
        chunk_overlap: int = 50,
    ) -> None:
        """基于给定的每页文本构建 GME 文本嵌入索引。

        仅负责：
        - 记录 self.page_texts
        - 文本分块（RecursiveCharacterTextSplitter）
        - 计算 GME 文本嵌入并构建 chunk-level 索引
        """
        self.page_texts = page_texts

        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_chunks: List[str] = []
        chunk2page: List[int] = []
        for page_idx, text in enumerate(page_texts):
            for chunk in splitter.split_text(text or ""):
                all_chunks.append(chunk)
                chunk2page.append(page_idx)

        if not all_chunks:
            self.chunk_embeddings = None
            self.chunk2page = None
            return

        with torch.no_grad():
            embeddings = self.gme.get_text_embeddings(all_chunks).cpu()

        self.chunk_embeddings = embeddings  # [n_chunks, dim]
        self.chunk2page = chunk2page

    def build_index(
        self,
        images: List[Image.Image],
        page_texts: Optional[List[str]] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 50,
    ):
        """构建基于 GME 文本嵌入的简单索引（不在内部执行 OCR）。

        推荐流程：
        1. 先通过 dots_ocr.parser 解析 PDF，得到 xxx_page_*.md
        2. 调用 extract_texts_from_ocr_dir(...) 得到 page_texts，并生成纯文本 md
        3. 再调用本方法或 build_index_from_texts(page_texts, ...)
        """
        if page_texts is None:
            # 未提供文本时，不构建索引
            self.page_texts = None
            self.chunk_embeddings = None
            self.chunk2page = None
            return

        self.build_index_from_texts(page_texts, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def get_page_text(self, page_idx: int) -> Optional[str]:
        if self.page_texts is None:
            return None
        if 0 <= page_idx < len(self.page_texts):
            return self.page_texts[page_idx]
        return None

    def search(self, query: str, top_k: int = 20) -> List[Dict]:
        """基于 GME 文本嵌入进行简单的文本到文本检索"""
        if self.chunk_embeddings is None or self.chunk2page is None:
            return []

        with torch.no_grad():
            query_emb = self.gme.get_text_embeddings([query]).cpu()[0]  # [dim]

        chunk_emb = self.chunk_embeddings  # [n_chunks, dim]
        scores = (chunk_emb * query_emb).sum(-1).numpy()
        order = np.argsort(scores)[::-1]

        page_best_scores: Dict[int, float] = {}
        for idx in order:
            page_idx = int(self.chunk2page[idx])
            if page_idx not in page_best_scores:
                page_best_scores[page_idx] = float(scores[idx])
            if len(page_best_scores) >= top_k:
                break

        sorted_pages = sorted(page_best_scores.items(), key=lambda x: x[1], reverse=True)
        results: List[Dict] = []
        for rank, (page_idx, score) in enumerate(sorted_pages):
            results.append(
                {
                    "page_idx": int(page_idx),
                    "page_num": int(page_idx) + 1,
                    "score": float(score),
                    "rank": rank + 1
                }
            )
        return results

class VQAModel:
    """多模态问答模型（支持 API 调用）"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model_name: str = "gpt-4o",
        use_api: bool = True
    ):
        """
        初始化 VQA 模型
        
        Args:
            api_key: API 密钥
            api_base: API 基础 URL
            model_name: 模型名称
            use_api: 是否使用 API（True）或本地模型（False）
        """
        self.use_api = use_api
        self.model_name = model_name
        
        if use_api:
            # 使用 API
            self.api_key = api_key or os.getenv("ARK_API_KEY")
            self.api_base = api_base or os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
            
            # 初始化 OpenAI 客户端
            self.client = OpenAI(
                base_url=self.api_base,
                api_key=self.api_key
            )
            
            print(f"✓ VQA Model initialized with API: {model_name}")
        else:
            # 使用本地模型（例如 Qwen2-VL）
            raise NotImplementedError("本地模型加载功能待实现，建议使用 API 模式避免显存不足")
    
    def answer_question(
        self,
        query: str,
        image: Image.Image,
        max_tokens: int = 512
    ) -> str:
        """
        基于单张图片回答问题
        
        Args:
            query: 问题
            image: PIL 图片
            max_tokens: 最大生成 token 数
            
        Returns:
            答案文本
        """
        if self.use_api:
            return self._answer_with_api(query, [image], max_tokens)
        else:
            return self._answer_with_local_model(query, [image], max_tokens)
    
    def answer_with_multiple_images(
        self,
        query: str,
        images: List[Image.Image],
        max_tokens: int = 512
    ) -> str:
        """
        基于多张图片综合回答问题（推荐用于 reranking 后的 top-k 结果）
        
        Args:
            query: 问题
            images: PIL 图片列表（例如 reranking top-5）
            max_tokens: 最大生成 token 数
            
        Returns:
            答案文本
        """
        if self.use_api:
            return self._answer_with_api(query, images, max_tokens)
        else:
            return self._answer_with_local_model(query, images, max_tokens)
    
    def _answer_with_api(self, query: str, images: List[Image.Image], max_tokens: int) -> str:
        """使用 API 回答问题（支持多图片）"""
        import base64
        from io import BytesIO
        
        # 构建 content 列表
        content = []
        
        # 添加问题文本
        if len(images) > 1:
            # 多图片时，添加提示语
            content.append({
                "type": "text",
                "text": f"请综合分析以下 {len(images)} 张图片，回答问题：{query}。若图片中没有符合关于query的内容，请回答不存在相关内容，不要编造。"
            })
        else:
            content.append({
                "type": "text",
                "text": query
            })
        
        # 添加所有图片
        for idx, image in enumerate(images):
            # 转换图片为 base64
            buffer = BytesIO()
            image.save(buffer, format="JPEG")
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_base64}"
                }
            })
        
        # 使用 OpenAI 客户端调用 API
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"API 请求失败: {str(e)}")
    
    def _answer_with_local_model(self, query: str, images: List[Image.Image], max_tokens: int) -> str:
        """使用本地模型回答问题（待实现）"""
        raise NotImplementedError("本地模型功能待实现")
    
    def answer_batch(
        self,
        queries: List[str],
        images: List[Image.Image],
        max_tokens: int = 512
    ) -> List[str]:
        """
        批量回答问题（每个查询对应单张图片）
        
        Args:
            queries: 问题列表
            images: 图片列表
            max_tokens: 最大生成 token 数
            
        Returns:
            答案列表
        """
        answers = []
        
        for query, image in tqdm(zip(queries, images), total=len(queries), desc="VQA Processing"):
            try:
                answer = self.answer_question(query, image, max_tokens)
                answers.append(answer)
            except Exception as e:
                print(f"Error processing query '{query}': {e}")
                answers.append(f"Error: {str(e)}")
        
        return answers
    
    def answer_batch_with_multiple_images(
        self,
        queries: List[str],
        all_images_list: List[List[Image.Image]],
        max_tokens: int = 512
    ) -> List[str]:
        """
        批量回答问题（每个查询对应多张图片，用于综合分析）
        
        Args:
            queries: 问题列表
            all_images_list: 每个查询对应的图片列表
                例如: [[img1_1, img1_2, img1_3], [img2_1, img2_2], ...]
            max_tokens: 最大生成 token 数
            
        Returns:
            答案列表
        """
        answers = []
        
        for query, images in tqdm(zip(queries, all_images_list), total=len(queries), desc="VQA Processing (Multi-Image)"):
            try:
                answer = self.answer_with_multiple_images(query, images, max_tokens)
                answers.append(answer)
            except Exception as e:
                print(f"Error processing query '{query}': {e}")
                answers.append(f"Error: {str(e)}")
        
        return answers

class MultiRoutePDFPipeline:
    """
    单 PDF 文档上的多路召回 Pipeline：
    1. ColQwen2-7B 文本到图像检索
    2. GME layout 级文本到图像检索
    3. GME 文本到文本检索（可选，可基于 OCR 文本）
    4. 使用 Qwen2.5-VL (ContentJudger) 对 ColQwen 结果做相关性过滤
    5. 将三路结果做加权融合
    """

    def __init__(
        self,
        device_index: int = 0,
        colqwen_path: Optional[str] = None,
        gme_path: Optional[str] = None,
        enable_text_route: bool = True,
        use_ocr: bool = True,
        colqwen_weight: float = 0.4,
        layout_weight: float = 0.3,
        text_weight: float = 0.3,
    ):
        self.device_index = device_index
        self.device = f"cuda:{device_index}" if torch.cuda.is_available() else "cpu"

        # ColQwen2-7B
        self.colqwen_retriever = ColQwenPDFRetriever(colqwen_path=colqwen_path)

        # GME 模型（layout + text 共用）
        if gme_path is None:
            gme_path = os.environ.get("GME_PATH")
        if not gme_path:
            raise ValueError("GME 模型路径未设置，请传入 gme_path 或设置环境变量 GME_PATH")
        self.gme_model = GmeQwen2VL(model_path=gme_path, device=self.device)

        self.layout_retriever = GMELayoutPDFRetriever(self.gme_model)
        self.enable_text_route = enable_text_route
        self.text_retriever = (
            GMETextPDFRetriever(self.gme_model, use_ocr=use_ocr) if enable_text_route else None
        )

        self.images: Optional[List[Image.Image]] = None

        # 路径权重
        self.colqwen_weight = colqwen_weight
        self.layout_weight = layout_weight
        self.text_weight = text_weight

        # Qwen2.5-VL 内容判断器，用于对 ColQwen 结果做 yes/no 过滤
        # self.judger = ContentJudger()

        # 图像重排序模型（MonoVLM），用于对 ColQwen 结果进行重排序
        self.reranker = ImageReranker(device=self.device)

    def build_index(self, images: List[Image.Image], page_texts: Optional[List[str]] = None):
        """为当前 PDF 构建所需的索引/缓存"""
        self.images = images
        self.layout_retriever.build_index(images)
        if self.text_retriever is not None:
            self.text_retriever.build_index(images, page_texts=page_texts)

    @staticmethod
    def _normalize_scores(scores_by_page: Dict[int, float]) -> Dict[int, float]:
        if not scores_by_page:
            return {}
        max_score = max(scores_by_page.values())
        if max_score <= 0:
            return {k: 0.0 for k in scores_by_page}
        return {k: v / max_score for k, v in scores_by_page.items()}

    def _rerank_colqwen_with_qwen(
        self,
        query: str,
        colq_results: List[Dict],
    ) -> List[Dict]:
        """使用 Qwen2.5-VL 对 ColQwen 初筛结果进行 yes/no 过滤"""
        # if not colq_results or self.images is None:
        """使用图像重排序模型 (MonoVLM) 对 ColQwen 初筛结果进行重排序"""
        if not colq_results or self.images is None or self.reranker is None:    
            return colq_results

        # reranked: List[Dict] = []
        # for item in colq_results:
        #     page_idx = int(item["page_idx"])
        #     image = self.images[page_idx]
        #     ocr_text = None
        #     if self.text_retriever is not None:
        #         ocr_text = self.text_retriever.get_page_text(page_idx)

        #     result, _ = self.judger.judge_relevance(image, question=query, ocr_text=ocr_text)

        #     new_item = dict(item)
        #     new_item["vl_relevance"] = result
        #     # 对判断为 "no" 的页面进行降权
        #     if result == "yes":
        #         new_item["rerank_score"] = float(item["score"])
        #     else:
        #         new_item["rerank_score"] = float(item["score"]) * 0.1
        #     reranked.append(new_item)

        # reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        # for rank, item in enumerate(reranked, 1):
        #     item["rank"] = rank

         # 准备候选页面的原始图片列表（顺序与 colq_results 对齐）
        candidate_images: List[Image.Image] = [
            self.images[int(item["page_idx"])] for item in colq_results
        ]

        # 使用 MonoVLM 对所有候选页面进行重排序
        # 这里 top_k 取 len(colq_results)，保证对所有候选都打分
        rerank_results = self.reranker.rerank(
            query=query,
            images=candidate_images,
            top_k=len(candidate_images),
        )

        # 根据重排序结果重新组织 ColQwen 结果，并记录 rerank_score
        reranked: List[Dict] = []
        for rr in rerank_results:
            idx = int(rr["doc_id"])  # 在 ImageReranker 中，doc_id 即候选列表下标
            base_item = colq_results[idx]
            new_item = dict(base_item)
            new_item["rerank_score"] = float(rr["score"])
            new_item["rank"] = int(rr["rank"])
            reranked.append(new_item)

        return reranked

    def _merge_routes(
        self,
        colq_results: List[Dict],
        layout_results: List[Dict],
        text_results: List[Dict],
        final_top_k: int,
    ) -> List[Dict]:
        """将三路召回结果做简单的加权融合"""
        page_ids = set()
        colq_scores: Dict[int, float] = {}
        layout_scores: Dict[int, float] = {}
        text_scores: Dict[int, float] = {}
        # colq_relevance: Dict[int, str] = {}

        for item in colq_results:
            idx = int(item["page_idx"])
            page_ids.add(idx)
            # 若存在 rerank_score，则优先使用重排序分数；否则退回原始 ColQwen 分数
            score = float(item.get("rerank_score", item["score"]))
            colq_scores[idx] = score
            # if "vl_relevance" in item:
            #     colq_relevance[idx] = item["vl_relevance"]

        for item in layout_results:
            idx = int(item["page_idx"])
            page_ids.add(idx)
            layout_scores[idx] = float(item["score"])

        for item in text_results:
            idx = int(item["page_idx"])
            page_ids.add(idx)
            text_scores[idx] = float(item["score"])

        colq_norm = self._normalize_scores(colq_scores)
        layout_norm = self._normalize_scores(layout_scores)
        text_norm = self._normalize_scores(text_scores)

        fused: List[Dict] = []
        for idx in page_ids:
            final_score = (
                self.colqwen_weight * colq_norm.get(idx, 0.0)
                + self.layout_weight * layout_norm.get(idx, 0.0)
                + self.text_weight * text_norm.get(idx, 0.0)
            )
            fused.append(
                {
                    "page_idx": idx,
                    "page_num": idx + 1,
                    "final_score": float(final_score),
                    "colqwen_score": float(colq_scores.get(idx, 0.0)),
                    "gme_layout_score": float(layout_scores.get(idx, 0.0)),
                    "gme_text_score": float(text_scores.get(idx, 0.0)),
                    # "vl_relevance": colq_relevance.get(idx),
                }
            )

        fused.sort(key=lambda x: x["final_score"], reverse=True)
        return fused[:final_top_k]

    def retrieve(
        self,
        queries: List[str],
        top_k_colqwen: int = 20,
        top_k_layout: int = 20,
        top_k_text: int = 20,
        final_top_k: int = 10,
        rerank_colqwen: bool = True,
    ) -> List[Dict]:
        """对一组查询执行三路召回 + ColQwen rerank + 多路融合"""
        if isinstance(queries, str):
            queries = [queries]
        if self.images is None:
            raise ValueError("请先调用 build_index(images) 为当前 PDF 构建索引")

        all_results: List[Dict] = []
        for query in queries:
            # 1) ColQwen text-to-image 检索
            colq_raw = self.colqwen_retriever.search(query, self.images, top_k=top_k_colqwen)
            if rerank_colqwen:
                colq_results = self._rerank_colqwen_with_qwen(query, colq_raw)
            else:
                colq_results = colq_raw

            # 2) GME layout-level text-to-image 检索
            layout_results = self.layout_retriever.search(query, top_k=top_k_layout)

            # 3) GME text-to-text 检索（可选）
            if self.text_retriever is not None:
                text_results = self.text_retriever.search(query, top_k=top_k_text)
            else:
                text_results = []

            # 4) 多路召回融合
            merged_results = self._merge_routes(
                colq_results=colq_results,
                layout_results=layout_results,
                text_results=text_results,
                final_top_k=final_top_k,
            )

            all_results.append(
                {
                    "query": query,
                    "colqwen_results": colq_raw,
                    "colqwen_reranked_results": colq_results,
                    "gme_layout_results": layout_results,
                    "gme_text_results": text_results,
                    "merged_results": merged_results,
                }
            )

        return all_results
