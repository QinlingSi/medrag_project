# scripts/07_retrieval_pipeline.py
import sys
import os
sys.path.append(os.path.dirname(__file__))

from importlib import import_module

qp = import_module('04_query_processing')
mpr_module = import_module('05_multipath_retrieval')
rr_module = import_module('06_reranker')


class MedRAGPipeline:
    def __init__(self, chunks_path, chroma_db_path, reranker_model_path=None):
        print("初始化 MultiPathRetriever...")
        self.retriever = mpr_module.MultiPathRetriever(
            chunks_path=chunks_path,
            chroma_db_path=chroma_db_path,
        )
        print("初始化 Reranker...")
        reranker_kwargs = {"model_name": reranker_model_path} if reranker_model_path else {}
        self.reranker = rr_module.MultiCriteriaReranker(**reranker_kwargs)

        # 建一个chunk_id -> (pub_date, journal) 的查找表，方便补充metadata
        self._meta_lookup = self.retriever.df.set_index('chunk_id')[['pub_date', 'journal']].to_dict('index')

    def _attach_metadata(self, candidates):
        for cand in candidates:
            meta = self._meta_lookup.get(cand['chunk_id'], {})
            cand['pub_date'] = meta.get('pub_date')
            cand['journal'] = meta.get('journal')
        return candidates

    def run(self, query, top_k_vector=20, top_k_keyword=20,
             fusion_strategy='rrf', top_k_fusion=20, top_k_final=10):
        """
        完整检索流水线：query理解 -> 多路检索融合 -> 补充metadata -> 重排序
        """
        query_info = qp.process_medical_query(query)

        candidates = self.retriever.fusion_search(
            query_info,
            top_k_vector=top_k_vector,
            top_k_keyword=top_k_keyword,
            fusion_strategy=fusion_strategy,
            top_k_final=top_k_fusion,
        )

        candidates = self._attach_metadata(candidates)

        final_results = self.reranker.rerank(
            query_info['cleaned_query'],
            candidates,
            top_k=top_k_final,
        )
        return final_results


if __name__ == "__main__":
    pipeline = MedRAGPipeline(
        chunks_path="../data/processed/chunks.parquet",
        chroma_db_path="../data/processed/chroma_db",
        reranker_model_path="../bge-reranker-base-cache",
    )
    results = pipeline.run("What is the effect of metformin on cardiovascular disease?")
    for r in results:
        print(r['rank'], r['chunk_id'], r['final_score'], r['text'][:60])
