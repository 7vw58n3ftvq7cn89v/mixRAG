import requests
import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain.chat_models import init_chat_model

load_dotenv()

class SiliconLLM:
    def __init__(self,api_key:str, model:str="Qwen/QwQ-32B"):
        self.model = model
        self.api_key = api_key
        self.url = "https://api.siliconflow.cn/v1/chat/completions"

    def chat(self, prompt:str):

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        response = requests.request("POST", self.url, json=payload, headers=headers)

        # 获取token_counts:{"prompt_tokens":28,"completion_tokens":2347,"total_tokens":2375}
        token_counts = response.usage

        return response.text.choices[0].message.content


if __name__ == "__main__":
    llm = SiliconLLM(api_key=os.getenv("SILICONFLOW_API_KEY"), model="deepseek-ai/DeepSeek-V2.5")
    print(llm.chat("What opportunities and challenges will the Chinese large model industry face in 2025?"))
