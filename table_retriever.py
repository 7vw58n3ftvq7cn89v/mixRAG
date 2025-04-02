import os
from typing import Optional, List, Any
from collections import Counter

import numpy as np
import pandas as pd
from langchain.docstore.document import Document
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
import json


class Retriever:
    def __init__(
            self,
            embed_model_name:str='local_models/m3e-base'
        ):
        self.wikidata_path = 'data/wikidata/'
        self.wikitable_schema_retriever = None
        self.schema_db_path = 'data/wikidata/{qid}/schemaDB'
        self.local_model_path = 'data/local_models/'
        self.embed_model_name = embed_model_name

        self.embedder = HuggingFaceEmbeddings(
            model_name=self.embed_model_name,
            model_kwargs={
                'trust_remote_code': True,
                'local_files_only': True  # 强制使用本地文件
            },
            cache_folder=self.local_model_path  # 指定缓存目录
        )


    def init_retriever(self):
        pass

    def retrieve_schema(self, qid:str, query:str):
        """查询相关schema，返回table_id和schema_name"""
        retriever = self.get_retriever(qid)
        results = retriever.invoke(query)
        table_list = [{'table_id': result.metadata['table_id'], 'schema_name': result.page_content} for result in results]
        return table_list

        
    
    def build_schema_corpus(self, qid:str):
        """将对应qid的schema进行embedding，存储在qid同一文件夹下
        存储形式：
            page_content:schema_name
            metadata: 
                table_id
                schema_text
        """
        table_path = os.path.join(self.wikidata_path, f'{qid}/tables.json')
        tables = json.load(open(table_path))
        docs = []
        for table in tables:
            table_id = table['table_id']
            schema_name = table['table_caption']
            schema_text = table['table_headers']
            for schema_name in table['table_headers']:
                metadata = {
                    'table_id': table_id,
                    'schema_text': schema_text
                }
                docs.append(Document(page_content=schema_name, metadata=metadata))
        return docs
    
    
    def get_retriever(self, qid:str):
        db_path = os.path.join(self.wikidata_path, f'{qid}/schemaDB')
        if not os.path.exists(db_path):
            # create db
            docs = self.build_schema_corpus(qid)
            db = FAISS.from_documents(docs, self.embedder)
            db.save_local(db_path)
        # load db
        db = FAISS.load_local(db_path, self.embedder, allow_dangerous_deserialization=True)
        # retrieve
        return db.as_retriever(search_kwargs={'k': 5})
    

class reference_retriever:

    def get_retriever(self, data_type, table_id, df):
        docs = None
        if self.mode == 'embed' or self.mode == 'hybrid':
            db_dir = os.path.join(self.db_dir, f'{data_type}_db_{self.max_encode_cell}_' + table_id)
            if os.path.exists(db_dir):
                if self.verbose:
                    print(f'Load {data_type} database from {db_dir}')
                db = FAISS.load_local(db_dir, self.embedder, allow_dangerous_deserialization=True)
            else:
                docs = self.get_docs(data_type, df)
                db = FAISS.from_documents(docs, self.embedder)
                db.save_local(db_dir)
            embed_retriever = db.as_retriever(search_kwargs={'k': self.top_k})
        if self.mode == 'bm25' or self.mode == 'hybrid':
            if docs is None:
                docs = self.get_docs(data_type, df)
            bm25_retriever = BM25Retriever.from_documents(docs)
            bm25_retriever.k = self.top_k
        if self.mode == 'hybrid':
            # return EnsembleRetriever(retrievers=[embed_retriever, bm25_retriever], weights=[0.9, 0.1])
            return EnsembleRetriever(retrievers=[embed_retriever, bm25_retriever], weights=[0.5, 0.5])
        elif self.mode == 'embed':
            return embed_retriever
        elif self.mode == 'bm25':
            return bm25_retriever

    def get_docs(self, data_type, df):
        if data_type == 'schema':
            return self.build_schema_corpus(df)
        elif data_type == 'cell':
            return self.build_cell_corpus(df)
        elif data_type == 'row':
            return self.build_row_corpus(df)
        elif data_type == 'column':
            return self.build_column_corpus(df)

    def build_schema_corpus(self, df):
        docs = []
        for col_name, col in df.items():
            if col.dtype != 'object' and col.dtype != str:
                result_text = f'{{"column_name": "{col_name}", "dtype": "{col.dtype}", "min": {col.min()}, "max": {col.max()}}}'
            else:
                most_freq_vals = col.value_counts().index.tolist()
                example_cells = most_freq_vals[:min(3, len(most_freq_vals))]
                result_text = f'{{"column_name": "{col_name}", "dtype": "{col.dtype}", "cell_examples": {example_cells}}}'
            docs.append(Document(page_content=col_name, metadata={'result_text': result_text}))
        return docs