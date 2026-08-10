# scripts/download_reranker.py —— 一次性下载bge-reranker-base到本地缓存，跑完可删除
import os
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["HTTP_PROXY"] = "http://127.0.0.1:6594"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:6594"

from transformers import AutoTokenizer, AutoModelForSequenceClassification

print("下载tokenizer...")
AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
print("下载模型...")
AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-base")
print("完成，模型已缓存到本地")
