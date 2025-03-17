import json
import time
import requests
from bs4 import BeautifulSoup
import ollama
from nano_graphrag._utils import wrap_embedding_func_with_attrs
# 如果你要使用 GraphRAG/QueryParam，则保留；如果不需要 RAG，可忽略。
# from nano_graphrag import GraphRAG, QueryParam  
#(可以替换为其他embedding方法)
import numpy as np
import os
import re
import pandas as pd
import csv

EMBEDDING_MODEL = "bge-large"
EMBEDDING_MODEL_DIM = 1024
EMBEDDING_MODEL_MAX_TOKENS = 8192

WIKI_URL = "https://en.wikipedia.org/wiki/{}"
WIKI_URL_QID = "https://www.wikidata.org/wiki/{}"

TABLE_SAVE_PATH = "tables.jsonl"
INFOBOX_SAVE_PATH = "infoboxes.json"

@wrap_embedding_func_with_attrs(
    embedding_dim=EMBEDDING_MODEL_DIM,
    max_token_size=EMBEDDING_MODEL_MAX_TOKENS,
)
async def ollama_embedding(texts: list[str]) -> np.ndarray:
    embed_text = []
    for text in texts:
        data = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
        embed_text.append(data["embedding"])
    return embed_text

def qid_to_entity_name(qid):
    response = requests.get(f"https://www.wikidata.org/wiki/{qid}")
    if response.status_code != 200:
        return None
        
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # 方法1：从页面标题获取
    title = soup.find('title')
    if title:
        # 移除 "- Wikidata" 后缀
        name = title.text.replace(' - Wikidata', '').strip()
        return name
    
    # 方法2：从meta标签获取
    meta_title = soup.find('meta', property='og:title')
    if meta_title:
        return meta_title.get('content')
        
    # 方法3：从meta description获取
    meta_desc = soup.find('meta', property='og:description')
    if meta_desc:
        desc = meta_desc.get('content')
        # 通常描述格式为 "xxx (1946-2020)"，我们只要名字部分
        name = desc.split('(')[0].strip()
        return name
    
    return None

def format_entity_name_for_wikipedia(name: str) -> str:
    """将实体名称中的空格转为下划线，便于拼接 Wikipedia 链接。"""
    return name.replace(" ", "_")

def get_wikipedia_page(ent_dict, section: str = None) -> str:
    try:
        if ent_dict.get('name') and ent_dict['name'] != "Not Found!":
            entity_name = format_entity_name_for_wikipedia(ent_dict['name'])
        elif ent_dict['id'] != 'None':
            # 如果没有实际的函数，可自行实现 QID -> Wikipedia 标题 的逻辑
            qid = ent_dict['id']
            entity_name = format_entity_name_for_wikipedia(qid_to_entity_name(qid))
            if entity_name is None:
                return "Not Found!"
        else:
            return "Not Found!"

        if entity_name == "Not Found!":
            return "Not Found!"
        else:
            wikipedia_url = f'https://en.wikipedia.org/wiki/{entity_name}'
            print('wikipedia_url:', wikipedia_url)

            response = requests.get(wikipedia_url, headers={'Connection': 'close'}, timeout=180)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")
            content_div = soup.find("div", {"id": "bodyContent"})

            # 移除脚本和样式
            for script_or_style in content_div.find_all(["script", "style"]):
                script_or_style.decompose()

            # 如果指定了 section，就只抓取该章节
            if section:
                header = content_div.find(
                    lambda tag: tag.name == "h2" and section in tag.get_text()
                )
                if header:
                    content = ""
                    for sibling in header.find_next_siblings():
                        if sibling.name == "h2":
                            break
                        content += sibling.get_text()
                    return content.strip()
                else:
                    return f"Section '{section}' not found."

            # 否则默认返回词条开头部分
            summary_content = ""
            for element in content_div.find_all(recursive=False):
                # 遇到第一个 h2 就认为简介结束
                if element.name == "h2":
                    break
                summary_content += element.get_text()
            return summary_content.strip()
    except Exception as e:
        print("Error fetching Wikipedia page:", e)
        return "Not Found!"

def main_old():
        # 1) 读取 JSON 文件，并取前 100 条
    with open("WebQSP.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # 不需要 RAG 的逻辑时，你可以直接跳过此处。
    # 这里只是演示如何初始化，如果真的不需要，就注释掉。
    # rag = GraphRAG(
    #     working_dir="./cwq-top100",
    #     enable_naive_rag=True,
    #     embedding_func=ollama_embedding,
    # )

    # 用于保存最终找到的 wiki 信息
    found_data = []
    cnt=0
    # 2) 对前 100 条数据的 qid_topic_entity 逐一下载维基页面
    for idx, item in enumerate(data):
        if(cnt>=100):
            print("Already collected 100 questions that have wiki_page")
            break
        question_id = item.get("QuestionId", f"unknown_{idx}")
        question_str = item.get("RawQuestion",{})
        qid_topic_entity = item.get("qid_topic_entity", {})
        
        # 如果没有 qid_topic_entity，跳过即可
        if not qid_topic_entity:
            continue
        cnt+=1
        for qid, entity_label in qid_topic_entity.items():
            if not entity_label:
                continue

            # 组织一个 ent_dict 给 get_wikipedia_page
            ent_dict = {
                "id": qid,
                "name": entity_label
            }
            wiki_text = get_wikipedia_page(ent_dict)

            if wiki_text != "Not Found!":
                # 记录下来当前问题ID+实体+wiki内容
                found_data.append({
                    "question_id": question_id,
                    "qid": qid,
                    "question_str":question_str,
                    "entity_label": entity_label,
                    "wiki_page": wiki_text
                })
                time.sleep(2)

    # 如果你想要将下载到的信息写回文件（JSON），可以执行以下操作：
    with open("found_wiki_data_webqsp.json", "w", encoding="utf-8") as fw:
        json.dump(found_data, fw, ensure_ascii=False, indent=2)
    print(f"写入成功，共找到 {len(found_data)} 条维基信息。")

    # 如果不需要 RAG 逻辑，可到此结束；下面是示例如何把文本插入到 RAG 里：
    # wiki_text_list = [item["wiki_page"] for item in found_data]
    # rag.insert(wiki_text_list)
    # result = rag.query("What are the top themes in this story?", param=QueryParam(mode="naive"))
    # print(result)

def is_infobox(table:BeautifulSoup):
    """判断是否是infobox"""
    return table.name == 'table' and table.get('class') == ['infobox']

def parse_infobox(table:BeautifulSoup, save_path:str=INFOBOX_SAVE_PATH):
    """解析维基百科的infobox表格"""
    info_dict = {}
    
    # 获取标题
    caption = table.find('caption', class_='infobox-title')
    if caption:
        info_dict['title'] = caption.text.strip()
    
    # 遍历所有行
    for row in table.find_all('tr'):
        # 获取标签（表头）
        label = row.find('th', class_='infobox-label')
        # 获取数据
        data = row.find('td', class_='infobox-data')
        
        # 处理子标题行
        header = row.find('th', class_='infobox-header')
        if header:
            current_section = header.text.strip()
            info_dict[current_section] = {}
            continue
            
        if label and data:
            label_text = label.text.strip()
            
            # 处理数据单元格中的特殊情况
            data_text = ""
            
            # 处理列表
            if data.find('ul'):
                items = [li.text.strip() for li in data.find_all('li')]
                data_text = items
            else:
                # 处理普通文本，移除引用标记
                data_text = ' '.join(
                    text.strip() for text in data.stripped_strings 
                    if not text.startswith('[') and not text.endswith(']')
                )
            
            # 如果在某个section下，则添加到对应的字典中
            if 'current_section' in locals():
                info_dict[current_section][label_text] = data_text
            else:
                info_dict[label_text] = data_text
                
    with open(save_path, "a", encoding="utf-8") as f:
        json.dump(info_dict, f, ensure_ascii=False, indent=2)
    
    return info_dict

def get_tables_from_page(url:str)->tuple[list, list[pd.DataFrame]]:

    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    # content_div = soup.find("div", {"id": "bodyContent"})
    # # 移除脚本和样式
    # for script_or_style in content_div.find_all(["script", "style"]):
    #     script_or_style.decompose()


    tables_soup = soup.find_all("table")
    infoboxes = []
    wikitables = []
    
    # 清空原有文件
    with open(INFOBOX_SAVE_PATH, "w", encoding="utf-8") as f:
        f.write("")
    with open(TABLE_SAVE_PATH, "w", encoding="utf-8") as f:
        f.write("")

    # parse infoboxes and wikitables
    for i,table_soup in enumerate(tables_soup):
        if is_infobox(table_soup):
            info_dict = parse_infobox(table_soup)
            infoboxes.append(info_dict)
        else:
            caption, headers, rows = parse_wikitable(table_soup)
            table_dict = wikitable_to_json(f"{i}_{caption}", headers, rows, save=True)
            wikitables.append(table_dict)
    return infoboxes, wikitables

def wikitable_to_json(caption:str, headers:list[str], rows:list[list[str]], save_path:str=TABLE_SAVE_PATH, save:bool=True):
    """把table保存为json格式"""
    table_dict = {
        "table_caption": caption,
        "table_headers": headers,
        "table_text": headers + rows
    }

    if save:
        with open(save_path, "a", encoding="utf-8") as f:
            json.dump(table_dict, f, ensure_ascii=False, indent=2)
    return table_dict


def parse_wikitable(table:BeautifulSoup):
    """html中提取表格元素"""
    # 获取caption
    caption = table.find('caption')
    if caption:
        caption_text = caption.text.strip()
    else:
        caption_text = ""
    
    # 获取表头，第一行tr
    headers = [th.text.strip() for th in table.find('tr').find_all('th')]

    # 获取数据行
    rows = []
    for tr in table.find_all('tr')[1:]:  # 跳过表头行
        row = []
        for td in tr.find_all('td'):
            # 处理可能的链接
            if td.find('a'):
                cell_value = td.find('a').text.strip()
            else:
                cell_value = td.text.strip()
            row.append(cell_value)
        if row:  # 确保行不为空
            rows.append(row)
    
    return caption_text, headers, rows

def main():
    # 示例使用
    url = "https://en.wikipedia.org/wiki/The_World%27s_Billionaires"
    infoboxes, wikitables = get_tables_from_page(url)
    
    # 打印每个表格的基本信息
    # for i, df in enumerate(dataframes):
    #     print(f"\n表格 {i+1} 的信息:")
    #     print(f"形状: {df.shape}")
    #     print(f"列名: {df.columns.tolist()}")
    #     print("\n前几行数据:")
    #     print(df.head())

if __name__ == "__main__":
    main()