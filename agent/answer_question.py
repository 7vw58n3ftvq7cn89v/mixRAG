import os
from pathlib import Path
from dotenv import load_dotenv
import json
import pandas as pd
from agent.rag_agent import TableRAGAgent
from agent.model import Model

# 加载环境变量
load_dotenv()

TABLE_NAME = "KT_discussion"
QUESTION_1 = """
请教下各位大佬，我目前使用4090部署KT 0.2.2b版本，运行DS R1 671B Q4量化版，最终的生成速度是 8.4 token每秒，在当前的配置下还有提升的空间吗？

CPU：INTEL(R) XEON(R) GOLD 6530 双路 64核
内存：512GB，带宽150GB/ss

"""

QUESTION_2 = """
这个配置能跑多少token？请估计一下：
Intel i7 4.2 boost 4c8t
32gb ddr4 ram trident Z rgb
RTX nividia 3070 8GB
Coolermaster cpu cooler with rgb
"""

def generate_table_json(table_caption:str, question:str):
    """生成用于RAG的表格数据结构"""
    df = pd.read_csv(f"data/{table_caption}.csv")
    columns = df.columns.tolist()
    values = df.values.tolist()
    table_text = [columns] + values

    table_json = {
        "table_id": table_caption.replace(' ', '_'),
        "question": question,
        "table_caption": table_caption,
        "table_text": table_text
    }

    return table_json

def tableRAG_answer(
        question:str,
        table_caption:str,
        model_name = 'deepseek-ai/DeepSeek-V2.5',
        provider = 'siliconflow',
        retrieve_mode = 'embed',
        embed_model_name = 'local_models/m3e-base',
        log_dir = 'output/qa_test',
        db_dir = 'db/',
        top_k = 5,
        max_encode_cell = 1000,
        verbose = False,
):
    """主函数：初始化RAG agent并回答问题"""
    # 创建必要的目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 准备agent参数
    task = 'qa'  # 设置为问答任务
    agent_args = {
        'model_name': model_name,
        'provider': provider,
        'retrieve_mode': retrieve_mode,
        'embed_model_name': embed_model_name,
        'task': task,
        'agent_type': 'TableRAG',
        'top_k': top_k,
        'max_encode_cell': max_encode_cell,
        'log_dir': log_dir,
        'db_dir': db_dir,
        'verbose': verbose
    }

    # 生成表格数据
    table_json = generate_table_json(table_caption, question)
    
    # 初始化agent并回答问题
    agent = TableRAGAgent(**agent_args)
    answer = agent.answer_question(table_json)
        
    return answer

def direct_answer(question:str, model_name:str="deepseek-ai/DeepSeek-V2.5", provider:str="siliconflow"):
    """直接回答问题"""
    model = Model(model_name=model_name,provider=provider)
    return model.query(question)

if __name__ == "__main__":
    print(f"问题：{QUESTION_2}")
    a = tableRAG_answer(question=QUESTION_2, table_caption=TABLE_NAME)
    b = direct_answer(question=QUESTION_2, model_name="deepseek-ai/DeepSeek-V2.5", provider="siliconflow")
    print(f"tableRAG_answer: {a}")
    print(f"direct_answer: {b}")
