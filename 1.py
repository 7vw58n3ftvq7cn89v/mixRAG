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

if __name__ == "__main__":
    basic_langchain_demo()
