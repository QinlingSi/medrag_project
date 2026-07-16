
# scripts/03_embed_and_index.py — 向量化与索引构建

import os
os.environ["HF_HUB_OFFLINE"] = "1"

import json
from datetime import datetime

import numpy as np
import pandas as pd
import chromadb
from FlagEmbedding import FlagModel

QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages:"
EMBEDDINGS_PATH = "../data/processed/chunk_embeddings.npy"
CHROMA_PATH = "../data/processed/chroma_db"
COLLECTION_NAME = "medrag_chunks"
STATS_PATH = "../data/processed/chroma_stats.json"


def init_embedding_model():
    return FlagModel(
        "BAAI/bge-small-en-v1.5",
        query_instruction_for_retrieval=QUERY_INSTRUCTION,
        use_fp16=True,
        devices=["mps"],
        batch_size=16,
    )


def build_metadata(record):
    def safe(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return ""
        return v
    return {
        "doc_id": safe(record["doc_id"]),
        "chunk_index": int(record["chunk_index"]),
        "total_chunks": int(record["total_chunks"]),
        "source_title": safe(record["source_title"]),
        "token_count": int(record["token_count"]),
        "pmid": safe(record["pmid"]),
        "journal": safe(record["journal"]),
        "pub_date": safe(record["pub_date"]),
    }


def query_index(collection, model, query_text, n_results=5, where_filter=None, max_safe_results=100):
    """
    Args:
        query_text: 查询文本
        model: 嵌入模型（FlagModel实例，用于编码query_text）
        n_results: 返回结果数量
        where_filter: 元数据过滤条件，例如 {"journal": "Nature"}
        max_safe_results: n_results的安全上限，避免超过ChromaDB/SQLite单次查询变量数限制
    Returns:
        查询结果（ChromaDB返回的dict，含ids/documents/metadatas/distances）
    """
    if n_results > max_safe_results:
        print(f"提示: 请求的n_results({n_results})超过安全上限({max_safe_results})，已自动调整为{max_safe_results}")
        n_results = max_safe_results

    query_embedding = model.encode_queries([query_text])[0].tolist()
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where_filter,
    )


if __name__ == "__main__":
    # 1. 加载chunk数据 
    chunks_df = pd.read_parquet("../data/processed/chunks.parquet")
    print(f"chunk总数: {len(chunks_df)}")

    #  2. 初始化模型 + 小样本验证 
    model = init_embedding_model()
    print("模型加载完成")

    sample_texts = chunks_df.sample(n=20, random_state=42)["text"].tolist()
    doc_embeddings = model.encode_corpus(sample_texts)
    print(f"文档向量维度: {doc_embeddings.shape}")

    test_query = "What is the effect of aspirin on cardiovascular disease?"
    query_embedding = model.encode_queries([test_query])
    print(f"查询向量维度: {query_embedding.shape}")

    # 3. 全量chunk向量：已存在就直接读，不重新计算 
    if os.path.exists(EMBEDDINGS_PATH):
        print(f"检测到已有向量文件，直接加载: {EMBEDDINGS_PATH}")
        all_embeddings = np.load(EMBEDDINGS_PATH)
    else:
        print(f"\n开始生成全部{len(chunks_df)}条chunk的向量...")
        all_texts = chunks_df["text"].tolist()
        all_embeddings = model.encode_corpus(all_texts)
        print(f"全部文档向量维度: {all_embeddings.shape}")
        np.save(EMBEDDINGS_PATH, all_embeddings)
        print(f"向量已保存到 {EMBEDDINGS_PATH}")

    assert len(chunks_df) == len(all_embeddings), "chunk数量和向量数量对不上"

    # 4. 创建持久化ChromaDB合集（余弦相似度）
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        print(f"检测到已存在的合集 {COLLECTION_NAME}，先删除重建，避免重复/残留数据")
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"合集已创建: {COLLECTION_NAME}（余弦相似度）")

    #  5. 构建id、文本、元数据，分批写入 
    ids = chunks_df["chunk_id"].astype(str).tolist()
    documents = chunks_df["text"].tolist()
    records = chunks_df.to_dict("records")
    metadatas = [build_metadata(r) for r in records]

    BATCH = 2000
    total = len(ids)
    for i in range(0, total, BATCH):
        end = min(i + BATCH, total)
        collection.add(
            ids=ids[i:end],
            embeddings=all_embeddings[i:end].tolist(),
            documents=documents[i:end],
            metadatas=metadatas[i:end],
        )
        print(f"已写入 {end}/{total}")

    #  6. 验证索引大小 
    count = collection.count()
    print(f"\n索引验证：collection.count() = {count}（预期 {total}）")
    assert count == total, "索引数量对不上，需要检查"

    #  7. 保存统计信息 
    stats = {
        "collection_name": COLLECTION_NAME,
        "total_chunks": total,
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "embedding_dimension": int(all_embeddings.shape[1]),
        "index_built_at": datetime.now().isoformat(),
        "chunk_size_stats": {
            "mean": round(float(chunks_df["token_count"].mean()), 1),
            "max": int(chunks_df["token_count"].max()),
            "min": int(chunks_df["token_count"].min()),
        } if "token_count" in chunks_df.columns else {},
        "metadata_fields": list(metadatas[0].keys()) if metadatas else [],
    }
    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"\n统计信息已保存到 {STATS_PATH}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    #  8. query函数demo验证
    demo_result = query_index(collection, model, test_query, n_results=3)
    print("\n=== query_index demo ===")
    for i, (doc, meta, dist) in enumerate(zip(
        demo_result["documents"][0], demo_result["metadatas"][0], demo_result["distances"][0]
    )):
        print(f"\n[{i+1}] 距离: {dist:.4f}")
        print(f"标题: {meta['source_title']}")
        print(f"片段: {doc[:150]}...")
