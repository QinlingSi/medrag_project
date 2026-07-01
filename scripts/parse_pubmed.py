import os
import xml.etree.ElementTree as ET
import chromadb

# 初始化 Chroma
client = chromadb.Client()
collection = client.create_collection("pubmed_medical")

xml_dir = "PMC000xxxxxx"
files = [f for f in os.listdir(xml_dir) if f.endswith(".xml")][:20]

docs = []
ids = []
skipped = 0

for fname in files:
    try:
        tree = ET.parse(os.path.join(xml_dir, fname))
        root = tree.getroot()

        # 提取标题
        title_el = root.find(".//article-title")
        title = title_el.text if title_el is not None else ""

        # 提取摘要
        abstract_el = root.find(".//abstract")
        abstract = " ".join(abstract_el.itertext()) if abstract_el is not None else ""

        if title and abstract:
            docs.append(f"{title}\n{abstract}")
            ids.append(fname.replace(".xml", ""))
        else:
            skipped += 1

    except Exception as e:
        skipped += 1

# 写入 Chroma
collection.add(documents=docs, ids=ids)
print(f"写入成功：{len(docs)} 篇，跳过：{skipped} 篇")

# 测试查询
results = collection.query(query_texts=["diabetes treatment"], n_results=3)
print("\n=== 查询 'diabetes treatment' 结果 ===")
for doc in results["documents"][0]:
    print("-", doc[:100])
