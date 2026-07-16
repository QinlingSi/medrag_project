#环境准备与数据加载

import json
import pandas as pd

# ---------- 1. 加载原始数据，转成DataFrame ----------
with open("../data/processed/dataset_expanded_clean.json", encoding="utf-8") as f:
    data = json.load(f)

df_raw = pd.DataFrame(data)
print(f"原始记录数: {len(df_raw)}")

# ---------- 2. 基础清洗 ----------
# abstract缺失的记录丢弃（依据设计说明：abstract缺失率8.36% > 1%阈值，且是核心检索内容，无法填充）
df_clean = df_raw[df_raw["abstract"].notna() & (df_raw["abstract"].str.strip() != "")].copy()
print(f"清洗后有效记录数: {len(df_clean)}（丢弃了 {len(df_raw) - len(df_clean)} 条abstract缺失的记录）")

# ---------- 3. 生成唯一标识 doc_id ----------
# 用 pmc_id 而不是 pmid：pmc_id完整率100%，pmid有9.08%缺失，不适合当主键
df_clean["doc_id"] = df_clean["pmc_id"]

assert df_clean["doc_id"].is_unique, "doc_id 存在重复，需要检查数据"
print(f"doc_id 唯一性检查通过，共 {df_clean['doc_id'].nunique()} 个唯一文献")


# 实施制定的文本分割策略

from transformers import AutoTokenizer
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

# ---------- 加载tokenizer ----------
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text, truncation=False))

# ---------- 初始化分割器 ----------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=450,
    chunk_overlap=80,
    length_function=count_tokens,
    separators=["\n\n", "\n", ". ", " ", ""],
    keep_separator="end",
)

def build_full_text(document) -> str:
    title = (document["title"] or "").rstrip(".")
    return f"{title}. {document['abstract']}"


def chunk_document(document) -> list[dict]:
    full_text = build_full_text(document)
    full_text_tokens = count_tokens(full_text)

    # ---------- 判断走哪个分支：token数是否超过chunk_size ----------
    if full_text_tokens > 450:
        # a. 根据长度进行智能分割
        texts = splitter.split_text(full_text)
        chunks = []
        for i, text in enumerate(texts):
            chunk_id = f"{document['doc_id']}_chunk{i}"
            chunk_data = {
                "chunk_id": chunk_id,
                "text": text,
                "doc_id": document["doc_id"],       # 归属的原文ID
                "chunk_index": i,                    # 块在原文中的序号
                "total_chunks": len(texts),          # 原文被分成的总块数
                "source_title": document["title"],   # 保留原文标题，便于追溯
                "token_count": count_tokens(text),
                "pmid": document.get("pmid"),
                "journal": document.get("journal"),
                "pub_date": document.get("pub_date"),
            }
            chunks.append(chunk_data)
        return chunks
    else:
        # b. 整体不分割
        data = {
            "chunk_id": document["doc_id"],   # 直接使用文献ID作为块ID
            "text": full_text,
            "doc_id": document["doc_id"],
            "chunk_index": 0,
            "total_chunks": 1,
            "source_title": document["title"],
            "token_count": full_text_tokens,
            "pmid": document.get("pmid"),
            "journal": document.get("journal"),
            "pub_date": document.get("pub_date"),
        }
        return [data]


# ---------- 对全部文献执行分块 ----------
all_chunks = []
for _, row in df_clean.iterrows():
    all_chunks.extend(chunk_document(row))

print(f"分块完成，共生成 {len(all_chunks)} 个chunk（来自 {len(df_clean)} 篇文献）")

# 分割文档，保存结果

import pandas as pd
import json
import os

# ---------- 保存chunk数据集 ----------
output_path = "../data/processed/chunks.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, ensure_ascii=False, indent=1)

print(f"chunk数据集已保存到 {output_path}")

# ---------- 保存处理配置和统计信息 ----------
chunks_df = pd.DataFrame(all_chunks)

stats = {
    "processed_date": pd.Timestamp.now().isoformat(),
    "original_documents": len(df_clean),
    "total_chunks": len(chunks_df),
    "chunks_per_doc": len(chunks_df) / len(df_clean) if len(df_clean) > 0 else 0,
    "chunk_size": 450,
    "chunk_overlap": 80,
    "embedding_model": "BAAI/bge-base-en-v1.5",
    "output_file": str(output_path),
    # 分支统计：整体不分割 vs 智能分割的文章各有多少
    "not_split_documents": int((chunks_df["total_chunks"] == 1).sum()),
    "split_documents": int(len(df_clean) - (chunks_df["total_chunks"] == 1).sum()),
    # chunk token数分布，供报告直接引用
    "chunk_token_distribution": {
        "mean": round(chunks_df["token_count"].mean(), 1),
        "min": int(chunks_df["token_count"].min()),
        "max": int(chunks_df["token_count"].max()),
        "p50": int(chunks_df["token_count"].quantile(0.5)),
        "p95": int(chunks_df["token_count"].quantile(0.95)),
    },
}

stats_path = "../data/processed/chunk_stats.json"
with open(stats_path, "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)

print(f"统计信息已保存到 {stats_path}")
print(json.dumps(stats, ensure_ascii=False, indent=2))
