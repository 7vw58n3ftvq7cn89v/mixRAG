from langchain.llms.base import LLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from openai import OpenAI
import os
import requests
import json
from typing import Any, List, Optional
from langchain.agents import initialize_agent, AgentType
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import BaseTool
from langchain_experimental.tools import PythonREPLTool
# 加载环境变量
load_dotenv()

class SiliconFlowLLM(LLM):
    api_key: str
    api_url: str = os.getenv("SILICONFLOW_API_BASE")
    model_name: str = os.getenv("QWEN_MODEL_NAME")
    @property
    def _llm_type(self) -> str:
        return "siliconflow"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        client = OpenAI(api_key=self.api_key, base_url=self.api_url)
        
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        if response.choices:
            return response.choices[0].message.content
        else:
            raise Exception(f"API调用失败: {response.text}")

class CustomCalculatorTool(BaseTool):
    name: str = "calculator"
    description: str = "用于进行基本数学计算，输入应该是数学表达式字符串"

    def _run(self, query: str) -> str:
        try:
            result = eval(query)
            return f"计算结果是: {result}"
        except:
            return "无法计算该表达式"

def basic_langchain_demo():
    # 初始化 SiliconFlow LLM
    llm = SiliconFlowLLM(api_key=os.getenv("SILICONFLOW_API_KEY"))
    
    # 初始化工具
    search = DuckDuckGoSearchRun()
    calculator = CustomCalculatorTool()
    
    tools = [
        search,
        calculator
    ]

    # 创建提示模板
    template = """
    你是一个有帮助的助手，可以使用多种工具来解决问题。

    可用工具:
    1. DuckDuckGo搜索: 用于在网络上搜索信息
    2. Python执行器: 用于执行Python代码
    3. 计算器: 用于进行数学计算

    历史对话:
    {chat_history}
    
    人类: {human_input}
    思考: 让我思考一下如何回答这个问题，是否需要使用工具。
    AI助手:"""
    
    prompt = PromptTemplate(
        input_variables=["chat_history", "human_input"],
        template=template
    )
    
    # 设置对话记忆
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # 创建agent
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        prompt=prompt,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True
    )
    
    # 示例对话
    print("第一次对话:")
    response1 = agent.run("计算 23 * 45 是多少？")
    print(response1)
    
    print("\n第二次对话:")
    response2 = agent.run("介绍数据湖")
    print(response2)


from table_retriever import Retriever
from prompts import get_prompt
from WikidataLoader import WikiDataLoader
from agent.model import Model
import re

class WikiAgent:
    def __init__(
            self,
            model_name,
            provider:str='siliconflow'
    ):
        self.model_name = model_name
        self.model = Model(model_name=model_name, provider='siliconflow')
        self.retriever = Retriever()
        self.dataloader = WikiDataLoader()

    def find_suitable_table(self,qid:str, query:str):
        if not os.path.exists(f'data/wikidata/{qid}'):
            self.dataloader.download_tables(qid)
        # 1.解析关键词
        col_names = self.extract_schema(query) #TODO
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
        print(f"found tables:{len(tables)}, sorted_tables: {sorted_tables}")
        for table_id, count in sorted_tables:
            # 判断是否可以回答问题
            judgement = self.judge_table(qid, table_id, query)
            if judgement:
                return table['table_id']
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
        

    def judge_table(self, qid:str, table_id:str, query:str):
        table_info = self.dataloader.load_table_info(qid=qid, table_id=table_id)
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

        print(f"table_id: {table_id}, thought: {thought}, answer: {answer}")
        
        # return {'thought':thought, 'answer':answer}
        if 'Yes' in answer:
            return True
        return False
    


if __name__ == "__main__":
    agent = WikiAgent(model_name=os.getenv('DS_MODEL_NAME'))
    query = "Who won the Best Actor award in the Los Angeles Film Critics Association Awards for the movie 'Taxi Driver'?"
    qid = 'Q47221'
    agent.find_suitable_table(qid=qid, query=query)

"""
1057,
"Who won the Best Actor award in the Los Angeles Film Critics Association Awards for the movie ""Taxi Driver""?",
movies,
"[{'id': 'Q47221', 'label': 'Taxi Driver'}]",
"[{'id': 'Q36949', 'label': 'Robert De Niro'}]",
table,Robert De Niro,352-2"""