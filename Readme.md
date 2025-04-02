# mixRAG
可以解析不同类型的数据作为知识库，支持text，table，infobox

## 项目介绍：写到简历
基于异构数据的RAG，可以解析不同类型的数据作为知识库，支持text，table，infobox



## Data 

### wikidata
按Qid分文件夹保存Wikipage的各类数据

实现功能：
qid to url

data extraction：
infobox
wikitable

### WikiDataLoader
[loader](./data/get_wiki_data.py)
- 主要功能：基于qid，给出entity name、infobox、wikitable、text
  - 提供从WikipediaPages解析text、infobox、wikitable的接口
  - 提供从qid获取entity name的接口
  - 提供从qid获取infobox、wikitable、text的接口
  - table: 提供表名、表描述、表数据

### TableAgent
[TableAgent](./agent/wiki_agent.py)
对于问题Q，检索相关表格，基于相关表格回答问题。

retrieval：
- schemaDB，对所有表格的schema进行embedding，向量存储
- 分解关键词，检索schemaDB，对相关表进行回答

### reference
[get_data.py](./data/get_data.py)
data type:
[page](https://en.wikipedia.org/wiki/The_World%27s_Billionaires)
[infobox](https://en.wikipedia.org/wiki/Template:Infobox)
[tables](https://www.mediawiki.org/wiki/Help:Tables)

[测试api：wiki api sandbox](https://en.wikipedia.org/wiki/Special:ApiSandbox#action=parse&format=json&page=Artificial%20intelligence&prop=sections&formatversion=2)

## TODO
- [ ] 用tablerag回答100个wiki问题
  - [x] 整理QA对，提取相应页面的表格
    - [x] 解析表格：解析页面得到wikitable和infobox索引，用pd.read_html获取表格，分别存储两类表格    
  - [ ] 用tablerag回答
    - [x] 搭建schemaDB，每个entity的所有tables
    - [x] Table retrieval的方法:检索schemaDB，判断表格是否可以回答问题
    - [ ] answer with table


如何基于Question找到相关表格？
- schemaDB + LLM判断
	- schemaDB，对每个table的每个列编码
	- LLM依据表格元数据，逐个判断是否可以回答问题（LLM需要注意判断表格能不能回答问题）
- table embeddings，合适的表格编码方法？ #wait 

## change log
2025-03-16: qid to url
2025-03-16: 从page中提取table
2025-03-16: 提取infobox
2025-03-16: 存储infobox和table
2025-03-22: 在wikipedia收集表格数据时，获取表名和表格描述的元数据，以便于后续的检索和推理
2025-03-27: 提取page的text数据
2025-03-29：下载实验数据
2025-03-30: 使用pd解析HTML页面，得到表格和infobox，解决表格不规范的问题