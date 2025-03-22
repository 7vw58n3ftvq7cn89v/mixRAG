# mixRAG
可以解析不同类型的数据作为知识库，支持text，table，infobox


## Data 

### wiki data collection
[get_data.py](./data/get_data.py)
data type:
[page](https://en.wikipedia.org/wiki/The_World%27s_Billionaires)
[infobox](https://en.wikipedia.org/wiki/Template:Infobox)
[tables](https://www.mediawiki.org/wiki/Help:Tables)

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




## TODO
- [ ]


## change log
2025-03-16: qid to url
2025-03-16: 从page中提取table
2025-03-16: 提取infobox
2025-03-16: 存储infobox和table
