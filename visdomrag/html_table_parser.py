# 第一步：HTML 清洗与入库 (HTML -> SQLite)
# OCR 出来的 HTML 表格通常很脏（列名重复、有空值），需要用 Pandas 清洗并存入一个临时的 SQLite 数据库。
import pandas as pd
from sqlalchemy import create_engine
import io

def html_table_to_sqlite(html_content, table_name, db_uri='sqlite:///rag_tables.db'):
    # 1. 使用 Pandas 解析 HTML
    # HTML 可能包含多个 table，我们取第一个，或者循环处理
    try:
        dfs = pd.read_html(io.StringIO(html_content))
        if not dfs:
            return False
        df = dfs[0]
    except Exception as e:
        print(f"解析 HTML 失败: {e}")
        return False

    # 2. 数据清洗 (至关重要)
    # 确保列名是字符串且没有特殊符号，方便 SQL 查询
    df.columns = [str(col).strip().replace(" ", "_").replace(".", "") for col in df.columns]
    
    # 3. 存入 SQLite
    engine = create_engine(db_uri)
    df.to_sql(table_name, engine, if_exists='replace', index=False)
    
    return True, list(df.columns)

# 模拟 OCR 出来的 HTML
ocr_html = """
<table>
  <thead>
    <tr><th>产品名称</th><th>2023销量</th><th>单价(元)</th></tr>
  </thead>
  <tbody>
    <tr><td>高性能显卡</td><td>5000</td><td>8999</td></tr>
    <tr><td>机械键盘</td><td>12000</td><td>499</td></tr>
  </tbody>
</table>
"""

# 执行入库
success, columns = html_table_to_sqlite(ocr_html, "sales_data_page1")
print(f"表已创建，列名: {columns}")

# 第二步：构建 Text2SQL RAG 模块
# 使用 LangChain 让 LLM 能够查询这个数据库。
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI

# 1. 连接到刚才创建的 SQLite 数据库
db = SQLDatabase.from_uri("sqlite:///rag_tables.db")

# 2. 初始化 LLM (建议使用 GPT-4 或 Qwen-Max 等逻辑强的模型)
llm = ChatOpenAI(model="gpt-4", temperature=0)

# 3. 创建 SQL Agent
# Agent 的作用是：理解问题 -> 看表结构 -> 写 SQL -> 运行 SQL -> 解释结果
agent_executor = create_sql_agent(
    llm=llm,
    db=db,
    agent_type="openai-tools",
    verbose=True
)

# 4. 测试提问
question = "机械键盘的销售总额是多少？" 
# 逻辑：LLM 会自动生成 SELECT 2023销量 * 单价 FROM sales_data_page1 WHERE 产品名称='机械键盘'
response = agent_executor.invoke(question)

print(f"最终回答: {response['output']}")

# 第三步：RAG 混合检索（进阶）
# 在实际 PDF RAG 项目中，文档既有文本又有表格。你需要一个路由（Router）。

# 元数据提取： 当你解析 PDF 时，将表格转为 SQL 表，并生成一段描述（例如：“这是关于2023年电子产品销量的表格”）。
# 向量存储：
# 文本段落 -> 存入向量库。
# 表格描述 -> 存入向量库（附带 Metadata: type='sql', table_name='sales_data_page1'）。
# 检索逻辑：


# 伪代码逻辑
def rag_pipeline(user_query):
    # 1. 在向量库中搜索相关内容
    docs = vector_store.similarity_search(user_query)
    
    context_text = ""
    sql_results = ""
    
    for doc in docs:
        if doc.metadata['type'] == 'text':
            context_text += doc.page_content
        elif doc.metadata['type'] == 'sql':
            # 2. 如果检索命中表格描述，触发 Text2SQL
            table_name = doc.metadata['table_name']
            # 指定 Agent 只查这张表
            specific_answer = agent_executor.invoke(f"Query table {table_name}: {user_query}")
            sql_results += specific_answer['output']
            
    # 3. 汇总回答
    final_prompt = f"""
    基于以下文本：{context_text}
    和以下数据库查询结果：{sql_results}
    回答用户问题：{user_query}
    """
    return llm.invoke(final_prompt)