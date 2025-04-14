import json
import time
import requests
from bs4 import BeautifulSoup
from io import StringIO
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

TABLE_SAVE_PATH = "wikidata/tables.jsonl"
INFOBOX_SAVE_PATH = "wikidata/infoboxes.json"


def get_wiki_text(entity_name):
    response = requests.get(
        'https://en.wikipedia.org/w/api.php',
        params={
            'action': 'query',
            'format': 'json',
            'titles': entity_name,
            'prop': 'extracts',
            'exintro': True,
            'explaintext': True,
        }).json()
        
    page = next(iter(response['query']['pages'].values()))
    return page['extract']


# 2. Get the full HTML of the page using the parse endpoint, parse it, and extract the first paragraph
# MediaWiki has a parse endpoint that you can hit with a URL like https://en.wikipedia.org/w/api.php?action=parse&page=Bla_Bla_Bla to get the HTML of a page. You can then parse it with an HTML parser like lxml (install it first with pip install lxml) to extract the first paragraph.

# For example:
def get_wiki_text2(entity_name):
    from lxml import html
    response = requests.get(
        'https://en.wikipedia.org/w/api.php',
        params={
            'action': 'parse',
            'page': 'Python_(programming_language)',
            'format': 'json',
        }).json()
    raw_html = response['parse']['text']['*']
    document = html.document_fromstring(raw_html)
    first_p = document.xpath('//p')[0]
    intro_text = first_p.text_content()
    print(intro_text)



class WikiDataLoader:
    def __init__(self):
        self.wikipage_url = "https://en.wikipedia.org/wiki/{}"
        self.wikidata_url = "https://www.wikidata.org/wiki/{}" # 使用qid
        self.local_wikidata_path = "data/wikidata/"
        self.data_path = "data/wikidata/{}"
        self.infobox_path = "data/wikidata/{}/infoboxes.json"
        self.table_path = "data/wikidata/{}/tables.json"
        self.table_path_from_pd = "data/wikidata/{}/tables_from_pd.json"
        self.text_path = "data/wikidata/{}/text.txt"
        self.qid_entity_path = "data/wikidata/qid_entity.csv"
        os.makedirs("data/wikidata", exist_ok=True)
    
    def load_wikidata(self, qid:str, url=None)->tuple[list, list]:
        """
        主函数，从qid获取infobox、table及其相关描述文本
        """
        text = self.get_wiki_text(qid)
        infoboxes, wikitables = self.load_tables(qid)
        return text, infoboxes, wikitables

    def load_tables(self, qid:str, url=None,from_pd:bool=True)->tuple[list, list]:
        """
        主函数，从qid获取infobox、table及其相关描述文本
        """
        infoboxes = []
        wikitables = []
        entity_name = self.qid_to_entity_name(qid)

        table_path = self.table_path_from_pd.format(qid) if from_pd else self.table_path.format(qid)
        infobox_path = self.infobox_path.format(qid)
        
        # 如果本地存在文件，直接读取
        if os.path.exists(infobox_path) and os.path.exists(table_path):
            try:
                with open(infobox_path, "r", encoding="utf-8") as f:
                    infoboxes = json.load(f)
            except Exception as e:
                print(f"Error loading infoboxes for qid: {qid}, error: {e}")
                infoboxes = []
            try:
                with open(table_path, "r", encoding="utf-8") as f:
                    wikitables = json.load(f)
            except Exception as e:
                print(f"Error loading wikitables for qid: {qid}, error: {e}")
                wikitables = []
            return infoboxes, wikitables
        else:
            success = self.download_tables(qid, url)
            if not success:
                raise Exception(f"Failed to download tables for qid: {qid}")
            return self.load_tables(qid, url)
    
    def download_tables(self, qid:str, url=None):
        """下载infobox和wikitable，包括从pd中下载"""
        infoboxes = []
        wikitables = []
        os.makedirs(self.data_path.format(qid), exist_ok=True)
        entity_name = self.qid_to_entity_name(qid)

        if url is None:
            url = self.wikipage_url.format(entity_name)

        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        
        tables_soup = soup.find_all("table")

        try:
            wikitables_indices = [i for i, table in enumerate(tables_soup) if 'wikitable' in ' '.join(table.get('class', ''))]
            all_tables_df = self.load_tables_by_pd(qid)
            # wikitables_df = all_tables_df[wikitables_indices]

            wikitables_from_pd = []
            for i,table_idx in enumerate(wikitables_indices):
                wikitable_df = all_tables_df[table_idx]
                table_soup = tables_soup[table_idx]
                headers = wikitable_df.columns.tolist()
                rows = wikitable_df.values.tolist()
                header_caption, table_description = self.get_table_description(table_soup)
                caption, _, _ = self.parse_wikitable(table_soup)
                if not caption:
                    # 如果caption为空，则使用标题作为caption
                    caption = header_caption
                table_dict = self.wikitable_to_json(
                    table_id=f"{qid}_{i}",
                    qid=qid, 
                    caption=f"{i}_{caption}", 
                    headers=headers, 
                    rows=rows, 
                    description=table_description
                )
                wikitables_from_pd.append(table_dict)
            with open(os.path.join(self.local_wikidata_path,f"{qid}/tables_from_pd.json"), "w", encoding="utf-8") as f:
                json.dump(wikitables_from_pd, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error downloading tables for qid: {qid}, error: {e}")
            return False

        # 找到所有表格
        wikitables_soup = soup.find_all("table", class_="wikitable")
        # infobox_soup = soup.find_all("table", class_=lambda x: x and "infobox" in x.lower())
        infobox_soup = soup.find_all("table", class_="infobox") 

        for i, infobox_soup in enumerate(infobox_soup):
            info_dict = self.parse_infobox(qid, infobox_soup)
            infoboxes.append(info_dict)
        
        for i, table_soup in enumerate(wikitables_soup):
            # 获取表格的描述文本
            header_caption, table_description = self.get_table_description(table_soup)
            caption, headers, rows = self.parse_wikitable(table_soup)
            if not caption:
                # 如果caption为空，则使用标题作为caption
                caption = header_caption
            table_dict = self.wikitable_to_json(
                table_id=f"{qid}_{i}",
                qid=qid, 
                caption=f"{i}_{caption}", 
                headers=headers, 
                rows=rows, 
                description=table_description
            )
            wikitables.append(table_dict)
        
        # 保存到本地
        with open(self.infobox_path.format(qid), "w", encoding="utf-8") as f:
            json.dump(infoboxes, f, ensure_ascii=False, indent=2)
        with open(self.table_path.format(qid), "w", encoding="utf-8") as f:
            json.dump(wikitables, f, ensure_ascii=False, indent=2)
        return True
    
    def load_tables_by_pd(self, qid:str):

        try:
            # 1. 获取正确编码的URL
            url = self.get_url_from_qid(qid)
                
            # 2. 发送请求获取HTML内容
            headers = {
                'User-Agent': 'Mozilla/5.0',
                'Accept-Charset': 'utf-8'
            }
            response = requests.get(url, headers=headers)
            response.encoding = 'utf-8'
            
            # 3. 使用HTML内容而不是URL来加载表格
            html_io = StringIO(response.text)
            tables = pd.read_html(html_io, encoding='utf-8')
            
            if not tables:
                print(f"未找到表格: {qid}")
                return []
                
            return tables
            
        except Exception as e:
            print(f"加载表格失败: {qid}, 错误: {e}")
            return []
    
    def load_single_table(self, qid:str, table_id, table_format:str='json'):
        if isinstance(table_id, str):
            table_id = int(table_id.split('_')[-1])
        table_path = self.table_path_from_pd.format(qid)
        tables = json.load(open(table_path))
        table = tables[table_id]
        if table_format == 'json':
            return table
        elif table_format == 'md':
            return self.table_to_md(table)
        else:
            raise ValueError(f"Invalid table format: {table_format}")
    
    def table_to_md(self, table:dict):
        """把table转换为md格式"""
        table_text = table['table_text']
        # 确保所有单元格都是字符串类型
        rows = [[str(cell) for cell in row] for row in table_text]
        return '\n'.join(['|'.join(row) for row in rows])
    
    def get_url_from_qid(self,qid):
        entity_name = self.qid_to_entity_name(qid)
        return self.wikipage_url.format(entity_name)
    
    def qid_to_entity_name_old(self,qid:str):
        # 如果本地存在，则直接读取
        df = pd.read_csv(self.qid_entity_path)
        if qid in df['qid'].values:
            return df[df['qid'] == qid]['entity'].values[0]
        
        # 如果本地不存在，则从wikidata.org获取
        response = requests.get(f"https://www.wikidata.org/wiki/{qid}")
        if response.status_code != 200:
            return None
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 方法1：从页面标题获取
        title = soup.find('title')
        if title:
            # 移除 "- Wikidata" 后缀
            name = title.text.replace(' - Wikidata', '').strip().replace(" ", "_")
            df = pd.concat([df, pd.DataFrame({'qid': [qid], 'entity': [name]})])
            df.to_csv(self.qid_entity_path, index=False)
            return name
        
        # 方法2：从meta标签获取
        meta_title = soup.find('meta', property='og:title')
        if meta_title:
            name = meta_title.get('content').replace(" ", "_")
            df = pd.concat([df, pd.DataFrame({'qid': [qid], 'entity': [name]})])
            df.to_csv(self.qid_entity_path, index=False)
            return name
            
        # 方法3：从meta description获取
        meta_desc = soup.find('meta', property='og:description')
        if meta_desc:
            desc = meta_desc.get('content')
            # 通常描述格式为 "xxx (1946-2020)"，我们只要名字部分
            name = desc.split('(')[0].strip().replace(" ", "_")
            df = pd.concat([df, pd.DataFrame({'qid': [qid], 'entity': [name]})])
            df.to_csv(self.qid_entity_path, index=False)
            return name
        
        return None
    
    def qid_to_entity_name(self, qid:str):
        """从qid获取实体名称"""
        
        # 如果本地存在，则直接读取
        df = pd.read_csv(self.qid_entity_path, encoding='utf-8')
        if qid in df['qid'].values:
            return df[df['qid'] == qid]['entity'].values[0]
        
        # 调用wiki api
        headers = {
            'User-Agent': 'Mozilla/5.0',
            'Accept-Charset': 'utf-8'
        }
        response = requests.get(
            'https://www.wikidata.org/w/api.php',
            params={
                'action': 'wbgetentities',
                'format': 'json',
                'ids': qid,
                'props': 'labels',
            },
            headers=headers
        ).json()
        if response['entities']:
            name = response['entities'][qid]['labels']['en']['value']
            name = name.replace(" ", "_")
            name = name.encode('utf-8').decode('utf-8')
            df = pd.concat([df, pd.DataFrame({'qid': [qid], 'entity': [name]})])
            df.to_csv(self.qid_entity_path, index=False, encoding='utf-8')
            return name
        else:
            print(f"No entity name for qid: {qid}")
            return None
    
    def parse_wikitable(self, table:BeautifulSoup):
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
            for td in tr.find_all(['td', 'th']):
                # 递归提取所有文本内容
                def extract_text(element):
                    if isinstance(element, str):
                        return element.strip()
                    elif hasattr(element, 'contents'):
                        return ''.join(extract_text(content) for content in element.contents)
                    return ''

                # 处理单元格内的所有内容
                cell_text = extract_text(td)
                # 清理多余的空白字符
                cell_text = ' '.join(cell_text.split())
                row.append(cell_text)
            
            if row:  # 确保行不为空
                rows.append(row)
        
        return caption_text, headers, rows

    def wikitable_to_json(self, table_id:int, qid:str, caption:str, headers:list[str], rows:list[list[str]], 
                         description:str="", save:bool=False):
        """把table保存为json格式"""
        
        table_dict = {
            "table_id": table_id,
            "qid": qid,
            "table_caption": caption,
            "table_headers": headers,
            "table_text": [headers] + rows,
            "table_description": description  # 添加描述文本
        }

        return table_dict

    def is_infobox(self, table:BeautifulSoup):
        """判断是否是infobox"""
        table_class = table.get('class', [])
        return table.name == 'table' and 'infobox' in table.get('class', [])

    def parse_infobox(self,qid:str,table:BeautifulSoup, save:bool=False):
        """解析维基百科的infobox表格"""
        save_path = self.infobox_path.format(qid)
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
                    
        # if save:
        #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
        #     with open(save_path, "a", encoding="utf-8") as f:
        #         json.dump(info_dict, f, ensure_ascii=False, indent=2)
        
        return info_dict

    def get_table_description(self, table_soup:BeautifulSoup)->tuple[str, str]:
        """
        获取表格的相关描述文本
        """
        description_texts = []
        caption = ""
        
        # 获取表格前面的所有相邻段落
        current = table_soup.find_previous_sibling()
        while current:
            # 遇到标题，记录为caption
            if current.name.startswith('h'):
                caption = current.text.strip()
                break
            # 如果遇到另一个表格或标题，停止搜索
            if current.find('table'):
                break
            
            # 获取段落文本，去除引用标记
            if current.name == 'p':  # 只处理段落标签
                text = ' '.join(
                    t.strip() for t in current.stripped_strings 
                    if not t.startswith('[') and not t.endswith(']')
                )
                
                if text:
                    description_texts.insert(0, text)  # 保持段落原有顺序
            
            elif current.name == 'div':
                if current.find('h2'):
                    caption = current.find('h2').text.strip()
                    break
                elif current.find('h3'):
                    caption = current.find('h3').text.strip()
                    break
            current = current.find_previous_sibling()
        
        
        return caption, '\n'.join(description_texts)
    
    def get_wiki_text(self,qid:str=None, entity_name:str=None):
        if os.path.exists(self.text_path.format(qid)):
            with open(self.text_path.format(qid), "r", encoding="utf-8") as f:
                return f.read()
        else:
            return self.download_wiki_text(qid, entity_name, save=True)
        
    
    def download_wiki_text(self, qid:str=None, entity_name:str=None, save=True):
        # 获取wiki文本
        if not entity_name:
            entity_name = self.qid_to_entity_name(qid)
        
        response = requests.get(
            'https://en.wikipedia.org/w/api.php',
            params={
                'action': 'query',
                'format': 'json',
                'titles': entity_name,
                'prop': 'extracts',
                # 'exintro': True,# 提取第一节
                'explaintext': True,
            }).json()
        page = next(iter(response['query']['pages'].values()))
        if 'extract' in page:
            wiki_text = page['extract']
        else:
            print(f"{entity_name} 没有提取到文本,page: {page}")
            wiki_text = ""
        
        # 确保目标文件夹存在
        os.makedirs(self.data_path.format(qid), exist_ok=True)
        
        if save:
            # 保存文本到文件
            with open(self.text_path.format(qid), "w", encoding="utf-8") as f:
                f.write(wiki_text)
        
        return wiki_text
    
    def load_table_info(self, qid:str, table_id:int):
        table_path = self.table_path_from_pd.format(qid)
        tables = json.load(open(table_path))
        table = tables[table_id]
        table_info = "table_caption: " + table['table_caption'] \
            + "\n" + "table_headers: " + str(table['table_headers']) 
        if table['table_description']:
            table_info += "\n" + "table_description: " + table['table_description']
        return table_info

def main():
    # 示例使用
    dataloader = WikiDataLoader()
    # url = "https://en.wikipedia.org/wiki/The_World%27s_Billionaires"
    # qid = "Q54935007"
    # # dataloader.download_tables(qid, url)
    # # infoboxes, wikitables = dataloader.load_tables(qid)
    # text = dataloader.get_wiki_text(qid=qid)
    # print(text)
    qid = "Q907568"
    _, wikitables = dataloader.load_tables(qid)
    print(wikitables)
    entity_name = dataloader.qid_to_entity_name(qid)
    print(entity_name.encode('utf-8').decode('utf-8'))
    

if __name__ == "__main__":
    main()