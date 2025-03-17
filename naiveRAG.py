import os
import requests
import hnswlib
import numpy as np
from typing import List, Tuple
import re
import json



naive_prompt = """
----Role----
You are a helpful agent tasked with using texts and any necessary common knowledge to answer a given question.
steps:

1.	You are given a question (Q) and a set of texts that may contain relevant information.
2.	Use the information from these texts, plus any necessary common knowledge, to answer the question.
3.	If the information provided is insufficient, clearly state that additional knowledge is required.
4.	If your answer includes the final result, enclose it in curly braces {}, if there are multiple answer seperate them with comma.
5.	Keep your explanation concise but sufficient to demonstrate how you arrived at your answer.


----Examples----

Example 1:

Q: what countries do people speak portuguese and is the country imports from Brazil?
texts:
ign in Japanese, Portuguese, and English in Oizumi, Japan, which has a large lusophone community due to the return immigration of Japanese Brazilians[43]
Portuguese is spoken by approximately 200 million people in South America, 30 million in Africa, 15 million in Europe, 5 million in North America and 0.
[96] Portuguese is a globalized language spoken officially on five continents, and as a second language by millions worldwide.
Since 1991, when Brazil signed into the economic community of Mercosul with other South American nations, namely Argentina, Uruguay and Paraguay, Portuguese is either mandatory, or taught, in the schools of those South American countries.

Main article: List of countries and territories where Portuguese is an official language
Countries and regions where Portuguese has official status
The Community of Portuguese Language Countries[6]
(in Portuguese Comunidade dos Países de Língua Portuguesa, with the Portuguese acronym CPLP) consists of the nine independent countries that have Portuguese as an official language: Angola, Brazil, Cape Verde, East Timor, Equatorial Guinea, Guinea-Bissau, Mozambique, Portugal and S?o Tomé and Príncipe.
[93] In such countries, Portuguese is spoken either as a native language by vast majorities due to their Portuguese colonial past or as a lingua franca in bordering and multilingual regions, such as on the Brazilian borders of Uruguay and Paraguay and in regions of Angola and Namibia. In many other countries, Portuguese is spoken by majorities as a second language.
 Standard European Portuguese is also the preferred standard by the Portuguese-speaking African countries. As such, and despite the fact that its speakers are dispersed around the world, Portuguese has only two dialects used for learning: the European and the Brazilian.
The official language of Brazil is Portuguese (Article 13 of the Constitution of the Federal Republic of Brazil), which almost all of the population speaks and is virtually the only language used in newspapers, radio, television, and for business and administrative purposes. Brazil is the only Portuguese-speaking nation in the Americas, making the language an important part of Brazilian national identity and giving it a national culture distinct from those of its Spanish-speaking neighbors.
Population of country[75][76][77]

More information

Native languageof the majority

Spoken by


 Brazil
203,062,512
Portuguese in Brazil

95% as a native language[78]


 Angola
35,981,281
Portuguese in Angola

40% as a native language, 60% total[79]


 Mozambique

 It is the official language of Angola, Brazil, Cape Verde, Guinea-Bissau, Mozambique, Portugal and S?o Tomé and Príncipe,[6] and has co-official language status in East Timor, Equatorial Guinea and Macau. Portuguese-speaking people or nations are known as Lusophone (lusófono).
 Brazil has the most Portuguese speakers in the world and is the only country in the Americas where Portuguese is an official language.[11][12]
Bounded by the Atlantic Ocean on the east, Brazil has a coastline of 7,491 kilometers (4,655 mi).[13] Covering roughly half of South America's land area, it borders all other countries and territories on the continent except Ecuador and Chile.
 The country is a federation composed of 26 states and a Federal District, which hosts the capital, Brasília. Its most populous city is S?o Paulo, followed by Rio de Janeiro. Brazil has the most Portuguese speakers in the world and is the only country in the Americas where Portuguese is an official language.

A: 
1. Countries where people speak Portuguese:
From the texts, Portuguese is spoken as an official language in the following countries:
	?	Angola
	?	Brazil
	?	Cape Verde
	?	East Timor
	?	Equatorial Guinea
	?	Guinea-Bissau
	?	Mozambique
	?	Portugal
	?	S?o Tomé and Príncipe

Additionally, Portuguese is used as a second language in several regions and is globally recognized.

2. Does each country import from Brazil?
The texts do not provide specific information on whether these countries import from Brazil. However, based on common knowledge:
	?	Many Lusophone countries (e.g., Angola, Portugal, and Mozambique) have strong economic ties with Brazil, including trade.
	?	Brazil is a significant exporter in the global market, making it likely that at least some of these countries import goods from Brazil.

Final answer:{The information provided is insufficient to answer the question}

Example 2:

Q: who fought in the battle of vicksburg and the country capital is Montgomery?
texts:
The siege of Vicksburg (May 18 – July 4, 1863) was the final major military action in the Vicksburg campaign of the American Civil War. In a series of maneuvers, Union Major General Ulysses S. Grant and his Army of the Tennessee crossed the Mississippi River and drove the Confederate Army of Mississippi, led by Lieutenant General John C.
Port Gibson
Raymond
Jackson
Champion Hill
Big Black River Bridge
Milliken's Bend
Lake Providence
Richmond
Goodrich's Landing
Helena
Vicksburg
Jackson expedition

The siege of Vicksburg (May 18 – July 4, 1863) was the final major military action in the Vicksburg campaign of the American Civil War.
 Pemberton, into the defensive lines surrounding the fortress city of Vicksburg, Mississippi, leading to the successful siege and Confederate surrender.
Vicksburg was the last major Confederate stronghold on the Mississippi River; therefore, capturing it completed the second part of the Northern strategy, the Anaconda Plan.
First Corinth
Booneville
Iuka
Second Corinth
Chickasaw Bayou
1863
Newton's Station
Grand Gulf
Snyder's Bluff
Port Gibson
Raymond
Jackson
Champion Hill
Big Black River Bridge
Vicksburg
1864
Aberdeen
Okolona
Yazoo City
Brice's Cross Roads
Oxford
Seminary
Senatobia

vteMississippi in the American Civil WarCombatants
Confederacy
State Troops
Union
Campaigns
Vicksburg
Yazoo Pass
Bayou
Greenville
Grierson's Raid
Jackson
Meridian
Yazoo City
Tupelo
Battles1862
First Corinth
Booneville
Iuka
Second Corinth
Chickasaw Bayou
1863

[49] This claim is inaccurate, for large Independence Day celebrations were held as early as 1907.[50]

Battlefield preservation[edit]
The works around Vicksburg are now maintained by the National Park Service as part of Vicksburg National Military Park. The park, located in Vicksburg, Mississippi, and Delta, Louisiana (flanking the Mississippi River), also commemorates the greater Vicksburg campaign which led up to the battle and includes reconstructed forts and trenches.
Armies
Cumberland
Georgia
Mississippi
Ohio
Tennessee

Mississippi
Tennessee

Authority control databases NationalUnited StatesCzech RepublicIsraelOtherNARA

Retrieved from https://en.wikipedia.org/w/index.php?title=Siege_of_Vicksburg&oldid=1265380246 Categories: 1863 in MississippiBattles of the Western Theater of the American Civil WarConflicts in 1863History of Vicksburg, MississippiHistory of Warren County, MississippiMay 1863 eventsJune 1863 eventsJuly 1863 eventsBattles of the American Civil War in MississippiSieges of the American Civil WarUnion victories of the American Civil WarVicksburg campaignBattles commanded by Ulysses S.
Coordinates: 32°20′37″N 90°51′04″W? / ?32.34361°N 90.85111°W? / 32.34361; -90.85111


From Wikipedia, the free encyclopedia
Battle of the American Civil War


Siege of VicksburgPart of the Vicksburg campaign of the Western Theater of the American Civil WarThe Siege of Vicksburg - Assault on Fort Hill by Thure de ThulstrupDateMay 18 – July 4, 1863(1 month, 2 weeks and 2 days)LocationWarren County, Mississippi32°20′37″N 90°51′04″W? / ?32.
The Siege of Vicksburg (May 18, 1863 – July 4, 1863) was a decisive Union victory during the American Civil War (1861–65) ..."; accessed June 2020

^ Historian Michael G. Ballard, in his Vicksburg campaign history, pp. 420–421, claims that this story has little foundation in fact.
title=Siege_of_Vicksburg&oldid=1265380246"Categories: 1863 in MississippiBattles of the Western Theater of the American Civil WarConflicts in 1863History of Vicksburg, MississippiHistory of Warren County, MississippiMay 1863 eventsJune 1863 eventsJuly 1863 eventsBattles of the American Civil War in MississippiSieges of the American Civil WarUnion victories of the American Civil WarVicksburg campaignBattles commanded by Ulysses S. GrantHidden categories: Pages using gadget WikiMiniAtlasArticles with short descriptionShort description is different from WikidataUse American English from April 2019All Wikipedia articles written in American EnglishUse mdy dates from April 2021Coordinates on WikidataArticles needing additional references from May 2021All articles needing additional referencesWebarchive template wayback linksPages using Sister project links with hidden wikidataGood articles

A: 
The Battle of Vicksburg during the American Civil War involved the Union forces, led by Major General Ulysses S. Grant, and the Confederate forces, led by Lieutenant General John C. Pemberton. The Union’s victory in this battle was a significant step in their Anaconda Plan, aimed at splitting the Confederacy by gaining control of the Mississippi River.

Montgomery is the capital of Alabama, which was part of the Confederacy during the Civil War. Therefore, the Confederacy, including Montgomery, was involved in the Battle of Vicksburg through its Confederate forces.

Finale answer: {Confederate States of America}


Example 3:

Q: what is the artist had a concert tour named I Am... World Tour 's daughters name
texts:
I Am... Yours(2009)

I Am... Tour (also referred to as the I Am... World Tour) was the fourth concert tour by American performer and singer-songwriter Beyoncé Knowles, in support of her third studio album, I Am... Sasha Fierce (2008), consisting of 110 concerts across six continents.
From Wikipedia, the free encyclopedia
(Redirected from I Am... World Tour)2009–2010 concert tour by Beyoncé


I Am... TourWorld tour by BeyoncéLocation
North America
Europe
Asia
Oceania
Africa
South America
Associated albumI Am... Sasha FierceStart dateMarch 26, 2009End dateFebruary 18, 2010No.
[60] It was certified double platinum by the Recording Industry Association of America (RIAA).[61] Another live CD/DVD of the tour was released as I Am... World Tour in late November 2010.[62] The album contained performances from different stops of the tour blended with personal footage of Knowles' backstage work and thoughts about the tour and her life.
 Retrieved December 1, 2009.

^ Sources about the North American leg of the I Am... World Tour:
Madison, Tjames (April 17, 2009). "Beyoncé lines up summer US dates for 'Sasha'". LiveDaily. Archived from the original on April 23, 2009. Retrieved May 10, 2009.

 It officially commenced in late April 2009, at Arena Zagreb in Croatia later visiting six continents, namely the Americas, Europe, Asia, Africa, and Australia.[9][10] The six-week North American leg of the I Am... tour kicked off on June 21 with a show at Madison Square Garden in New York and finished with a four-night residency at Encore in the Wynn Las Vegas from July 30, 2009 to August 2, 2009.
 The tour was announced in October 2008, and began in March 2009 with five preliminary ’rehearsal’ shows in North America. Knowles has described the I Am... World Tour as her best and most theatrical of all of her tours.
The set list for the tour included songs from Knowles' previous three studio albums, several covers of songs by other artists, as well as a Destiny's Child medley.
 She performed an unplugged, acoustic styled show different from the rest of her tour, at the Encore Theater in Las Vegas.[57] The show on August 2, 2009 was recorded and later released as a DVD, audio CD and television special in late November 2009 titled I Am...
 Archived from the original on June 16, 2012. Retrieved June 15, 2013.

^ a b c d e f g h i j k l m n o p Ganz, Caryn (June 22, 2009). "Beyonce Brings Hits, Jay-Z to "I Am..." Tour opener in New York". Rolling Stone. Archived from the original on February 3, 2013.
^ Music World Entertainment (2010) Beyoncé – I Am... World Tour (Liner Notes) Columbia Records.


External links[edit]



Wikimedia Commons has media related to I Am... Tour.

Beyoncé Knowles' official website
vteBeyoncé
Albums discography
Singles discography

^ Sources about tour dates for the Asian leg of the I Am... World Tour:
名古屋エリアのライブ、コンサート情報サイト. Tank! the Web (in Japanese). Sunday Folk Promotion. Archived from the original on July 24, 2009. Retrieved January 26, 2010.
"非我莫属"碧昂丝世界巡演北京演唱会-中国票务在线 [Beyoncé World Tour Beijing Concert].


A: 
To answer the question, we need to identify the daughter of Beyoncé Knowles, who had a concert tour named “I Am… World Tour.”

The texts do not explicitly mention Beyoncé’s daughter. However, using common knowledge, Beyoncé’s daughter is named Blue Ivy Carter.

Final answer: {Blue Ivy Carter}.

----Data----

"""


class LocalSentenceSplitter:
    def __init__(self, chunk_size, chunk_overlap) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @staticmethod
    def _zng(paragraph):
        pattern = u'([！？。…!?.\n])'
        return [sent for sent in re.split(pattern, paragraph, flags=re.U)]

    @staticmethod
    def _zng_new(paragraph):
        pattern = u'([！？。…!?.\n])'
        # return [sent for sent in re.split(pattern, paragraph, flags=re.U)]
        sentences = []
        sent_now = ''
        # 将句子与标点合并
        for s in re.split(pattern, paragraph, flags=re.U):
            if sent_now == '':
                sent_now = s
            elif len(s) <= len("！？"):
                sent_now += s
            else:
                sentences.append(sent_now)
                sent_now = s
        if sent_now != '':
            sentences.append(sent_now)
        return sentences

    def split_text(self, segment):
        chunks, chunk_now, size_now = [], [], 0
        no_left = False
        for s in LocalSentenceSplitter._zng_new(segment):
            no_left = False
            chunk_now.append(s)
            size_now += len(s)
            if size_now > self.chunk_size:
                chunk = "".join(chunk_now)
                chunk_now, size_now = self._get_overlap(chunk_now)
                chunks.append(chunk)
                no_left = True

        if no_left == False:
            chunks.append("".join(chunk_now))
        return chunks

    def _get_overlap(self, chunk):
        rchunk = chunk[:]
        rchunk.reverse()
        size_now, overlap = 0, []
        for s in rchunk[:-1]:
            overlap.append(s)
            size_now += len(s)
            if size_now > self.chunk_overlap:
                break
        overlap.reverse()
        return overlap, size_now

class OllamaEmbedding:
    """
    调用本地 Ollama 的 /embeddings 接口来获取文本的Embedding。
    假设 Ollama 版本支持 embeddings 功能，且在 localhost:11434 端口提供服务。
    你可以根据需要修改 URL、超时、模型名称等。
    """
    def __init__(self, host="http://localhost:11434", model="bge-large", work_dir="work_dir"):
        self.host = host
        self.model = model
        self.work_dir = work_dir
        # 日志文件，可在此记录一些embedding的调用信息
        self.log_path = os.path.join(self.work_dir, "logs.txt")
        os.makedirs(self.work_dir, exist_ok=True)

    def _log(self, msg: str):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        输入若干文本，返回它们的向量（shape=[len(texts), dim]）。
        """
        embeddings = []
        for text in texts:
            try:
                payload = {
                    "model": self.model,
                    "prompt": text
                }
                # Ollama 的 /embeddings 接口（注意版本不同，接口可能略有差异）
                url = f"{self.host}/api/embeddings"
                r = requests.post(url, json=payload, timeout=60)
                r.raise_for_status()
                data = r.json()
                # data 应该包含 "embedding" 字段，这里做个简单取值
                emb = data.get("embedding", [])
                embeddings.append(emb)
                self._log(f"[encode] success for text: {text[:30]}..., embedding dim={len(emb)}")
            except Exception as e:
                # 异常时可记录日志，返回一个零向量或者直接抛出
                self._log(f"[encode] error for text: {text[:30]}..., error={e}")
                # 这里简单用全0向量顶替
                embeddings.append([0.0]*1024)  
        return np.array(embeddings)


class HNSWVectorStore:
    """
    用 hnswlib 来管理向量索引。采用 'cosine' 作为距离度量。
    """
    def __init__(self, embedding_dim: int, work_dir: str = "work_dir"):
        self.embedding_dim = embedding_dim
        self.work_dir = work_dir
        os.makedirs(self.work_dir, exist_ok=True)

        self.index_path = os.path.join(self.work_dir, "index.bin")
        self.labels_path = os.path.join(self.work_dir, "labels.npy")

        # 判断索引文件是否已经存在，决定是加载还是新建
        if os.path.exists(self.index_path) and os.path.exists(self.labels_path):
            self.index = hnswlib.Index(space="cosine", dim=self.embedding_dim)
            self.index.load_index(self.index_path)
            self.index.set_ef(50)  # 设置检索效率
            loaded_labels = np.load(self.labels_path, allow_pickle=True)
            self.labels = loaded_labels.tolist() 
            print("[HNSWVectorStore] index loaded from disk.")
        else:
            # 初始化一个空索引
            self.index = hnswlib.Index(space="cosine", dim=self.embedding_dim)
            # 在插入前，需要先init一次
            # max_elements 需要设置足够大，避免后续超过限制
            self.index.init_index(max_elements=1000000, ef_construction=200, M=16)
            self.index.set_ef(50)
            self.labels = []
            print("[HNSWVectorStore] created new empty index.")

    def add_texts(self, texts: List[str], embeddings: np.ndarray):
        """
        将文本和其向量插入 HNSW 索引，并记录 labels（即文本内容）。
        """
        num_new = embeddings.shape[0]
        start_id = len(self.labels)
        ids = np.arange(start_id, start_id + num_new)

        # 插入向量
        self.index.add_items(embeddings, ids)
        # 记录文本，以便检索时能得到对应的文本
        self.labels.extend(texts)

    def save(self):
        """
        将索引和对应 labels 保存到磁盘。
        """
        self.index.save_index(self.index_path)
        np.save(self.labels_path, np.array(self.labels, dtype=object), allow_pickle=True)
        print("[HNSWVectorStore] index saved.")

    def similarity_search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[str]:
        """
        基于向量检索最相似的 Top-K 文本。
        """
        # hnswlib 返回的是 (ids, distances)
        ids, distances = self.index.knn_query(query_embedding, k=top_k)
        # ids shape: (1, top_k)   distances shape: (1, top_k)
        # 这里假设只有一个查询向量，所以取 ids[0]
        print(ids)
        result_texts = [self.labels[idx] for idx in ids[0]]
        return result_texts


class NaiveRAG:
    """
    一个简易的RAG示例：用本地 Ollama 生成Embedding，再用HNSW检索相似文本，最后拼接回答。
    """
    def __init__(self, embedding_dim: int = 1024, work_dir="work_dir", model="bge-large"):
        self.work_dir = work_dir
        self.embedding_model = OllamaEmbedding(
            host="http://localhost:11434",
            model=model,
            work_dir=work_dir
        )
        self.vector_store = HNSWVectorStore(embedding_dim=embedding_dim, work_dir=work_dir)

    def build_index(self, text_chunks: List[str]):
        """
        对text_chunks做embedding并插入索引，然后保存索引到磁盘。
        """
        embeddings = self.embedding_model.encode(text_chunks)  # [N, dim]
        self.vector_store.add_texts(text_chunks, embeddings)
        self.vector_store.save()

    def generate_prompt(self, query: str, top_k: int = 3) -> str:
        """
        对用户query做embedding，然后从索引中检索top_k文本块，用简单方式拼接。
        """
        query_emb = self.embedding_model.encode([query])[0]  # shape=[dim]
        retrieved_texts = self.vector_store.similarity_search(query_emb, top_k=top_k)
        #print(retrieved_texts)
        # 这里仅用一个很简单的拼接回答示例
        context = "\n".join(retrieved_texts)
        #answer = (
        #    f"You are a helpful assistant please answer the question {query} with given context:\n{context}"
        #)
        prompt = naive_prompt+f"Q:{query}\ntexts:{context}\nA:"
        return prompt

class TableRAG:
    def __init__(self):
        pass

    def build_index(self, text_chunks: List[str]):
        pass

    def generate_prompt(self, query: str, top_k: int = 3) -> str:
        pass