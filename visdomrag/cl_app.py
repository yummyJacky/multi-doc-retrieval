import os
# os.environ["VISDOM_PDF_PATHS"] = "/home/zechuan/m3docrag/contents/2024_Tencent_ESG.pdf,/home/zechuan/m3docrag/contents/2024_sanqi_ESG.pdf,/home/zechuan/m3docrag/contents/2024_architecture_ESG.pdf"
os.environ["VISDOM_LLM_MODEL"] = "doubao"
os.environ["QWEN_VL_SERVER_URL"] = "http://127.0.0.1:8001"
os.environ["QWEN_VL_MODEL"] = "Qwen/Qwen3-VL-4B-Instruct"
from pathlib import Path
from typing import List, Dict, Optional

import chainlit as cl

from visdom import VisDoMRAG
from dotenv import load_dotenv

load_dotenv()


def build_visdom_for_pdfs(pdf_paths: List[str]) -> VisDoMRAG:
    if not pdf_paths:
        raise ValueError("至少需要上传一个 PDF 文件。")

    llm_model = os.getenv("VISDOM_LLM_MODEL", "doubao")
    text_retriever = os.getenv("VISDOM_TEXT_RETRIEVER", "minilm")
    top_k = int(os.getenv("VISDOM_TOP_K", "5"))

    data_dir = str(Path(pdf_paths[0]).parent)
    output_dir = str(Path(data_dir) / "visdom_chainlit_output")
    os.makedirs(output_dir, exist_ok=True)

    api_keys: Dict[str, str] = {}
    doubao_key = os.getenv("ARK_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    if doubao_key:
        api_keys["doubao"] = doubao_key
    if openai_key:
        api_keys["openai"] = openai_key

    config: Dict[str, object] = {
        "data_dir": data_dir,
        "output_dir": output_dir,
        "llm_model": llm_model,
        "vision_retriever": "nemo",
        "text_retriever": text_retriever,
        "top_k": top_k * len(pdf_paths),
        "api_keys": api_keys,
        "force_reindex": False,
        "qa_prompt": "Answer the question based on the document text.",
        "pdf_files": pdf_paths,
        "csv_path": None,
        "ocr_engine": os.getenv("VISDOM_OCR_ENGINE", "dots"),
        "dots_max_completion_tokens": int(os.getenv("VISDOM_DOTS_MAX_TOKENS", "4096")),
        "dots_num_thread": int(os.getenv("VISDOM_DOTS_NUM_THREAD", "64")),
        "dots_dpi": int(os.getenv("VISDOM_DOTS_DPI", "200")),
        "dots_prompt_mode": os.getenv("VISDOM_DOTS_PROMPT_MODE", "prompt_layout_all_en"),
        "qwen_vl_server_url": os.getenv("QWEN_VL_SERVER_URL"),
        "qwen_vl_model": os.getenv("QWEN_VL_MODEL", "Qwen/Qwen3-VL-4B-Instruct"),
        "qwen_vl_api_key": os.getenv("QWEN_VL_API_KEY"),
        "text_use_rerank": os.getenv("VISDOM_TEXT_USE_RERANK", "false").lower() == "true",
    }

    return VisDoMRAG(config)


@cl.on_chat_start
async def on_chat_start():
    await cl.Message(
        content=(
            "欢迎使用 ESG VisDoMRAG 问答系统。\n\n"
            "请上传一个或多个 ESG 报告 PDF 文件，"
            "首次提问时必须附带 PDF 文件，后续提问则可直接输入问题。"
        ),
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    visdom: Optional[VisDoMRAG] = cl.user_session.get("visdom")
    if visdom is None:
        # 首次收到消息时，从附件中读取用户上传的 PDF
        elements = message.elements or []
        files = [e for e in elements if getattr(e, "mime", "").startswith("application/pdf")]

        if not files:
            await cl.Message(
                "首次使用请在附加一个或多个 ESG 报告 PDF 文件后再发送消息。"
            ).send()
            return

        pdf_paths: List[str] = []
        for f in files:
            if getattr(f, "path", None) and f.name.lower().endswith(".pdf"):
                pdf_paths.append(f.path)

        if not pdf_paths:
            await cl.Message("未找到有效的 PDF 文件，请确认上传的是 .pdf。" ).send()
            return

        try:
            visdom = await cl.make_async(build_visdom_for_pdfs)(pdf_paths)
        except Exception as e:
            await cl.Message(
                content=(
                    "初始化 VisDoMRAG 失败: "
                    f"{e}. 请检查服务器环境和上传的 PDF 是否有效。"
                )
            ).send()
            return

        cl.user_session.set("visdom", visdom)
        cl.user_session.set("pdf_paths", pdf_paths)

        await cl.Message(
            content="文档解析与索引已完成，现在可以开始提问。",
        ).send()

    question = message.content.strip()
    if not question:
        await cl.Message("请输入非空问题。" ).send()
        return

    await cl.Message("正在检索 ESG 报告并生成回答，请稍候……").send()

    answer_dict = await cl.make_async(visdom.answer_question)(question)

    if not answer_dict:
        await cl.Message("抱歉，未能基于当前文档回答该问题。请尝试换一种问法。" ).send()
        return

    answer = str(answer_dict.get("answer", "")).strip() or "(模型未返回明确答案)"
    analysis = str(answer_dict.get("analysis", "")).strip()
    conclusion = str(answer_dict.get("conclusion", "")).strip()

    parts = ["回答：" + answer]
    if conclusion:
        parts.append("结论：" + conclusion)
    if analysis:
        parts.append("分析：\n" + analysis)

    await cl.Message("\n\n".join(parts)).send()
