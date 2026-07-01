import chromadb

client = chromadb.Client()
collection = client.create_collection("test_medical")

collection.add(
    documents=[
        "心肌梗死是由冠状动脉阻塞引起的心肌细胞死亡",
        "2型糖尿病是胰岛素抵抗导致的血糖调节障碍",
        "高血压是指动脉血压持续升高的慢性病"
    ],
    ids=["doc1", "doc2", "doc3"]
)

results = collection.query(
    query_texts=["血糖问题"],
    n_results=2
)

print("=== 查询结果 ===")
for doc in results["documents"][0]:
    print("-", doc)
