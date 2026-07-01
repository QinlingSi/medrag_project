import json
import os

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from transformers import AutoTokenizer

# ---------- 1. 加载数据 ----------
with open("../data/processed/dataset.json", encoding="utf-8") as f:
    data = json.load(f)

valid = [r for r in data if r.get("abstract")]
print(f"有效记录数: {len(valid)}")

# ---------- 2. 加载真实tokenizer，定义token长度函数 ----------
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")

def token_len(text):
    return len(tokenizer.encode(text, truncation=False))

# ---------- 3. 配置分块器 ----------
# chunk_size/chunk_overlap 单位是“按length_function算出的长度”，这里指定用真实token数而不是字符数
splitter = RecursiveCharacterTextSplitter(
    chunk_size=450,
    chunk_overlap=80,
    length_function=token_len,
    separators=["\n\n", "\n", ". ", " ", ""],  # 优先按段落/句子切，实在不行才按空格硬切
)

# ---------- 4. 对全部文章执行分块 ----------
def chunk_documents(records):
    """输入：dataset.json里的记录列表
    输出：分块后的chunk列表，每个chunk带回原文章的元数据（journal/pub_date/pmid等），
    方便后续在向量库里做metadata filter和溯源"""
    all_chunks = []
    for r in records:
        full_text = f"{(r.get('title') or '').rstrip('.')}. {r['abstract']}"
        chunks = splitter.split_text(full_text)
        for i, chunk_text in enumerate(chunks):
            all_chunks.append({
                "chunk_id": f"{r['pmc_id']}_chunk{i}",
                "pmc_id": r["pmc_id"],
                "pmid": r.get("pmid"),
                "journal": r.get("journal"),
                "pub_date": r.get("pub_date"),
                "chunk_index": i,
                "total_chunks": len(chunks),
                "text": chunk_text,
                "token_count": token_len(chunk_text),
            })
    return all_chunks

all_chunks = chunk_documents(valid)
print(f"\n分块后总chunk数: {len(all_chunks)}（原始文章数: {len(valid)}）")

# ---------- 5. 验证结果 ----------
not_split = sum(1 for r in valid if len(splitter.split_text(f"{r.get('title') or ''}\n{r['abstract']}")) == 1)
print(f"未被分割（整篇当一个chunk）的文章数: {not_split} ({not_split/len(valid)*100:.1f}%)")

chunk_token_counts = [c["token_count"] for c in all_chunks]
over_limit_chunks = sum(1 for t in chunk_token_counts if t > 512)
print(f"分块后仍超过512 tokens的chunk数: {over_limit_chunks}（应该接近0）")

# 抽查一篇原本超长的文章，看分块效果
long_article = next(r for r in valid if token_len(f"{r.get('title') or ''}\n{r['abstract']}") > 600)
its_chunks = [c for c in all_chunks if c["pmc_id"] == long_article["pmc_id"]]
print(f"\n=== 抽查样例: {long_article['pmc_id']}，原文token数: {token_len(long_article['abstract'])} ===")
print(f"被切成 {len(its_chunks)} 个chunk:")
for c in its_chunks:
    print(f"  chunk{c['chunk_index']}: {c['token_count']} tokens — 开头: {c['text'][:80]}...")

# ---------- 6. 保存结果 ----------
output_path = "../data/processed/chunks.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, ensure_ascii=False, indent=1)
print(f"\n分块结果已保存到 {output_path}")