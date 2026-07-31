# scripts/05_multipath_retrieval.py
import os
os.environ["HF_HUB_OFFLINE"] = "1"
import pandas as pd
import jieba
from rank_bm25 import BM25Okapi
import chromadb
from FlagEmbedding import FlagModel

QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages:"


class MultiPathRetriever:
    def __init__(self, chunks_path, chroma_db_path, collection_name="medrag_chunks"):
        """
        chunks_path: data/processed/chunks.parquet 路径
        chroma_db_path: ChromaDB持久化目录
        """
        print("加载chunk数据...")
        self.df = pd.read_parquet(chunks_path)
        self.df = self.df.reset_index(drop=True)

        print("连接ChromaDB...")
        client = chromadb.PersistentClient(path=chroma_db_path)
        self.collection = client.get_collection(collection_name)

        print("加载embedding模型...")
        self.embed_model = FlagModel(
            "BAAI/bge-small-en-v1.5",
            query_instruction_for_retrieval=QUERY_INSTRUCTION,
            use_fp16=True,
            devices=["mps"],
        )

        print("构建BM25索引...")
        self._build_bm25_index()

    def _build_bm25_index(self):
        """用jieba分词 + BM25Okapi建索引，语料来自self.df['text']"""
        tokenized_corpus = [
            list(jieba.cut(text)) for text in self.df['text'].tolist()
        ]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.chunk_ids = self.df['chunk_id'].tolist()

    def vector_search(self, query_info, top_k=10):
        """
        向量检索路径
        注意：用cleaned_query而不是vector_query——
        vector_query已经手动拼了指令前缀，encode_queries()内部还会再加一次指令，
        会导致双重前缀，破坏向量质量
        """
        query_text = query_info['cleaned_query']
        query_embedding = self.embed_model.encode_queries([query_text])[0].tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )
        return self._format_chroma_results(results)

    def keyword_search(self, query_info, top_k=10):
        """
        BM25关键词检索路径
        """
        tokenized_query = list(jieba.cut(query_info['keyword_query']))
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = scores.argsort()[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            results.append({
                "chunk_id": self.chunk_ids[idx],
                "text": self.df.iloc[idx]['text'],
                "score": float(scores[idx]),
                "rank": rank + 1,
                "source": "bm25"
            })
        return results

    def _format_chroma_results(self, chroma_results):
        """把ChromaDB原始返回格式统一成和keyword_search一致的结构"""
        formatted = []
        ids = chroma_results['ids'][0]
        docs = chroma_results['documents'][0]
        distances = chroma_results['distances'][0]
        for rank, (cid, doc, dist) in enumerate(zip(ids, docs, distances)):
            formatted.append({
                "chunk_id": cid,
                "text": doc,
                "score": 1 - dist,
                "rank": rank + 1,
                "source": "vector"
            })
        return formatted



    def fusion_search(self, query_info, top_k_vector=10, top_k_keyword=10, 
                       fusion_strategy='rrf', top_k_final=10, rrf_k=60,
                       vector_weight=0.6, keyword_weight=0.4):
        """
        多路检索 + 融合
        Args:
            query_info: 04输出的查询信息
            top_k_vector: 向量检索候选数量
            top_k_keyword: 关键词检索候选数量
            fusion_strategy: 'rrf' / 'weighted' / 'simple'
            top_k_final: 融合后最终返回数量
            rrf_k: RRF公式里的常数k
            vector_weight/keyword_weight: weighted策略里的权重
        """
        vector_results = self.vector_search(query_info, top_k=top_k_vector)
        keyword_results = self.keyword_search(query_info, top_k=top_k_keyword)

        if fusion_strategy == 'simple':
            fused = self._fuse_simple(vector_results, keyword_results)
        elif fusion_strategy == 'rrf':
            fused = self._fuse_rrf(vector_results, keyword_results, rrf_k)
        elif fusion_strategy == 'weighted':
            fused = self._fuse_weighted(vector_results, keyword_results, vector_weight, keyword_weight)
        else:
            raise ValueError(f"未知融合策略: {fusion_strategy}")

        return fused[:top_k_final]

    def _fuse_simple(self, vector_results, keyword_results):
        vector_ids = {r['chunk_id']: r for r in vector_results}
        keyword_ids = {r['chunk_id']: r for r in keyword_results}
        both = set(vector_ids) & set(keyword_ids)
        only_vector = set(vector_ids) - both
        only_keyword = set(keyword_ids) - both

        both_sorted = sorted(both, key=lambda cid: vector_ids[cid]['rank'])
        only_v_sorted = sorted(only_vector, key=lambda cid: vector_ids[cid]['rank'])
        only_k_sorted = sorted(only_keyword, key=lambda cid: keyword_ids[cid]['rank'])

        fused = []
        for cid in both_sorted:
            r = vector_ids[cid].copy()
            r['source'] = 'both'
            fused.append(r)
        for cid in only_v_sorted:
            fused.append(vector_ids[cid])
        for cid in only_k_sorted:
            fused.append(keyword_ids[cid])

        for rank, r in enumerate(fused):
            r['rank'] = rank + 1
        return fused
    

    def _fuse_rrf(self, vector_results, keyword_results, k=60):
        """Reciprocal Rank Fusion: score = sum(1/(k+rank))"""
        scores = {}
        texts = {}
        for r in vector_results:
            cid = r['chunk_id']
            scores[cid] = scores.get(cid, 0) + 1 / (k + r['rank'])
            texts[cid] = r['text']
        for r in keyword_results:
            cid = r['chunk_id']
            scores[cid] = scores.get(cid, 0) + 1 / (k + r['rank'])
            texts[cid] = r['text']

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        fused = []
        for rank, (cid, score) in enumerate(ranked):
            fused.append({
                "chunk_id": cid,
                "text": texts[cid],
                "score": score,
                "rank": rank + 1,
                "source": "rrf"
            })
        return fused

    def _fuse_weighted(self, vector_results, keyword_results, vector_weight, keyword_weight):
        """归一化后加权求和"""
        def normalize(results):
            if not results:
                return {}
            scores = [r['score'] for r in results]
            lo, hi = min(scores), max(scores)
            span = hi - lo if hi > lo else 1
            return {r['chunk_id']: (r['score'] - lo) / span for r in results}

        v_norm = normalize(vector_results)
        k_norm = normalize(keyword_results)
        texts = {r['chunk_id']: r['text'] for r in vector_results + keyword_results}

        all_ids = set(v_norm) | set(k_norm)
        fused = []
        for cid in all_ids:
            score = vector_weight * v_norm.get(cid, 0) + keyword_weight * k_norm.get(cid, 0)
            fused.append({"chunk_id": cid, "text": texts[cid], "score": score, "source": "weighted"})

        fused.sort(key=lambda x: x['score'], reverse=True)
        for rank, r in enumerate(fused):
            r['rank'] = rank + 1
        return fused




