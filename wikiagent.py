
from table_retriever import Retriever
from prompts import get_prompt
from WikidataLoader import WikiDataLoader
from agent.model import Model
from agent import TableRAGAgent
from loguru import logger
import re
import pandas as pd
import json
import os
from typing import Optional

logger.add('logs/wiki_agent.log')

class WikiAgent:
    def __init__(
            self,
            model_name:str='deepseek-ai/DeepSeek-V2.5',
            provider:str='siliconflow'
    ):
        self.model_name = model_name
        self.model = Model(model_name=model_name, provider='siliconflow')
        self.retriever = Retriever()
        self.dataloader = WikiDataLoader()

    def find_suitable_table(self,qid:str, query:str) -> Optional[dict]:
        qid_path = f'data/wikidata/{qid}'
        if not os.path.exists(qid_path):
            self.dataloader.download_tables(qid)
        tables_path = os.path.join(qid_path, 'tables.json')
        tables = json.load(open(tables_path))
        if not tables:
            return None
        
        # 1.解析关键词
        col_names = self.extract_schema(query) 
        # 2.检索相关schema，得到table
        tables = {}
        for col_name in col_names:
            new_tables = self.retriever.retrieve_schema(qid, col_name) 
            for table in new_tables:
                if table['table_id'] not in tables:
                    tables[table['table_id']] = 1
                else:
                    tables[table['table_id']] += 1
        # 3.对每个table，按统计个数高到低顺序判断是否可以回答问题
        sorted_tables = sorted(tables.items(), key=lambda x: x[1], reverse=True)
        logger.info(f"found tables:{len(tables)}, sorted_tables: {sorted_tables}")
        for table_id, count in sorted_tables:
            # 判断是否可以回答问题
            table_id = int(table_id.split('_')[-1])
            judgement = self.judge_table(qid, table_id, query)
            if judgement:
                final_table = self.dataloader.load_single_table(qid, table_id)
                final_table['question'] = query
                return final_table
        return None
    
    def extract_schema(self, query:str):
        prompt = get_prompt(task='wiki', agent_type='Wikiagent', prompt_type='extract_schema_prompt', query=query)
        response = self.model.query(prompt)
        print(f'extract_schema: {response}')
        # pattern = r'\[(.*?)\]'
        # match = re.search(pattern, response)
        # schema = match.group(1).strip() if match else None
        schema = json.loads(response)
        
        # 确保返回的是列表
        if not isinstance(schema, list):
            return []
        
        return schema
        

    def judge_table(self, qid:str, table_id:int, query:str):
        """判断table是否可以回答问题"""
        table_info = self.dataloader.load_table_info(qid=qid, table_id=table_id)
        logger.debug(f"table_info: {table_info}")
        table_prompt = get_prompt(task='wiki', agent_type='Wikiagent', prompt_type='judge_table_prompt', table_info=table_info,query=query)
        response = self.model.query(table_prompt)
        # 提取Thought部分
        thought_pattern = r'<Thought>(.*?)</Thought>'
        thought_match = re.search(thought_pattern, response, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else None
        
        # 提取Answer部分
        answer_pattern = r'<Answer>(.*?)</Answer>'
        answer_match = re.search(answer_pattern, response, re.DOTALL)
        answer = answer_match.group(1).strip() if answer_match else None

        logger.debug(f"table_id: {table_id}, thought: {thought}, answer: {answer}")
        
        # return {'thought':thought, 'answer':answer}
        if 'Yes' in answer:
            return True
        return False
    

def tableRAG_answer(
        # question:str,
        # table_caption:str,
        table:dict,
        qid:str,
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
    
    # 初始化agent并回答问题
    agent = TableRAGAgent(**agent_args)
    answer = agent.answer_question(table)
        
    return answer

def convert_to_json(s):
    # 先处理字典的单引号，但保留词中的撇号
    # 使用正则表达式匹配字典的键值对中的单引号
    s = re.sub(r"'(id|label)':", r'"\1":', s)
    # 处理值中的单引号，但排除所有格撇号
    s = re.sub(r":\s*'([^']*)'", r': "\1"', s)
    return s

def wiki_qa(
        qa_path:str,
        model_name = 'deepseek-ai/DeepSeek-V2.5',
        provider = 'siliconflow',
        retrieve_mode = 'embed',
        embed_model_name = 'local_models/m3e-base',
        log_dir = 'data/logs/wikiqa',
        db_dir = 'data/db/wikidata',
        top_k = 5,
        max_encode_cell = 1000,
        verbose = False,
):
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

    wiki_agent = WikiAgent()
    rag_agent = TableRAGAgent(**agent_args)
    df = pd.read_csv(qa_path)
    df['agent_answer'] = None
    for i,row in df.iterrows():
        question = row['question']
        logger.debug(f"Round {i}, Current question: {question}")
        entities = json.loads(convert_to_json(row['entities']))
        for entity in entities:
            qid = entity['id']
            logger.debug(f"start finding table from qid: {qid}")
            table = wiki_agent.find_suitable_table(qid=qid, query=question)
            if table:
                table['question'] = question
                answer = rag_agent.answer_question(table)
                logger.debug(f"question: {question}, \ncorrect answer: {row['answer_text']}, \nrag answer: {answer}")
                df.at[i, 'agent_answer'] = answer
    df.to_csv(f'data/CompMix/tableqa_sample_with_answer.csv', index=False)
    return df

def main():

    agent = WikiAgent(model_name=os.getenv('DS_MODEL_NAME'))
    query = "Who won the Best Actor award in the Los Angeles Film Critics Association Awards for the movie 'Taxi Driver'?"
    qid = 'Q47221'
    table = agent.find_suitable_table(qid=qid, query=query)
    table['question'] = query
    answer = tableRAG_answer(
        table=table, 
        qid=qid,
        db_dir=f'data/db/wikidata',
        log_dir='data/logs/wikiqa'
    )
    print(answer)

if __name__ == "__main__":
    qa_path = 'data/CompMix/tableqa_sample.csv'
    wiki_qa(qa_path=qa_path)
    # main()