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
- 主要功能：基于qid，给出entity name、infobox、wikitable、text
  - 提供从WikipediaPages解析text、infobox、wikitable的接口
  - 提供从qid获取entity name的接口
  - 提供从qid获取infobox、wikitable、text的接口
  - table: 提供表名、表描述、表数据


### reference
[get_data.py](./data/get_data.py)
data type:
[page](https://en.wikipedia.org/wiki/The_World%27s_Billionaires)
[infobox](https://en.wikipedia.org/wiki/Template:Infobox)
[tables](https://www.mediawiki.org/wiki/Help:Tables)

[wiki api sandbox](https://en.wikipedia.org/wiki/Special:ApiSandbox#action=parse&format=json&page=Artificial%20intelligence&prop=sections&formatversion=2)

## TODO
- [ ] 用tablerag回答100个wiki问题


## change log
2025-03-16: qid to url
2025-03-16: 从page中提取table
2025-03-16: 提取infobox
2025-03-16: 存储infobox和table
2025-03-22: 在wikipedia收集表格数据时，获取表名和表格描述的元数据，以便于后续的检索和推理
2025-03-27: 提取page的text数据