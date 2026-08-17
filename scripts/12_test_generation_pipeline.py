"""
scripts/12_test_generation_pipeline.py

批量测试生成pipeline：跑一组不同类型的query，记录关键指标到reports/日志，
用于对比prompt效果、验证幻觉引用修复的稳定性。

每条query测试之间会清理一次GPU(MPS)缓存，避免连续多次调用embedding/reranker模型
导致显存耗尽（8GB统一内存机器上，连续几次调用后容易出现
"Insufficient Memory kIOGPUCommandBufferCallbackErrorOutOfMemory"报错，
进而污染检索结果）。
"""

import sys
import os
import json
import time
import gc

sys.path.append(os.path.dirname(__file__))
from importlib import import_module

generation_module = import_module("11_generation_pipeline")
MedicalGenerationPipeline = generation_module.MedicalGenerationPipeline

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


TEST_QUERIES = [
    "What is the effect of metformin on cardiovascular disease?",
    "What are the differences between metformin and sulfonylureas in treating type 2 diabetes?",
    "What are the common side effects of metformin?",
    "What is the recommended dosage of metformin for elderly patients?",
    "Can metformin be used together with insulin?",
]


def _clear_gpu_memory():
    """清理MPS缓存，避免连续多次调用embedding/reranker模型导致GPU内存耗尽"""
    gc.collect()
    if _TORCH_AVAILABLE and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def run_batch_test(
    pipeline: MedicalGenerationPipeline,
    queries: list,
    retrieval_pipeline=None,
    output_path: str = "../reports/generation_test_log.jsonl",
    top_k_final: int = 5,
):
    """
    对每条query依次跑生成pipeline，逐条追加写入jsonl日志文件（每行一个json对象）。

    Args:
        pipeline: 已初始化好的MedicalGenerationPipeline
        queries: 待测试的query列表
        retrieval_pipeline: 已初始化好的MedRAGPipeline，用于实时检索；
                             如果为None，则要求queries传入的是(query, retrieved_docs)元组列表
        output_path: 日志输出路径（jsonl格式，追加写入，不覆盖之前的记录）
        top_k_final: 检索返回条数
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    for i, query in enumerate(queries):
        print(f"\n========== [{i + 1}/{len(queries)}] {query} ==========")

        if retrieval_pipeline is not None:
            t0 = time.time()
            retrieved_docs = retrieval_pipeline.run(query, top_k_final=top_k_final)
            retrieval_time = time.time() - t0
        else:
            raise ValueError("需要提供retrieval_pipeline来实时检索，或改用预存的retrieved_docs")

        result = pipeline.run(query, retrieved_docs)

        log_entry = {
            "query": query,
            "answer": result.get("answer"),
            "answer_length_chars": len(result.get("answer") or ""),
            "citation_check": result.get("citation_check"),
            "generation_metrics": result.get("generation_metrics"),
            "retrieval_time_seconds": round(retrieval_time, 2),
            "sources": result.get("sources"),
            "error": result.get("error"),
            "timestamp": result.get("timestamp"),
        }

        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        # 简要打印，方便实时观察
        success = result.get("generation_metrics", {}).get("stage_success", {})
        has_halluc = (result.get("citation_check") or {}).get("has_hallucination")
        print(f"  answer长度: {log_entry['answer_length_chars']}字符")
        print(f"  stage_success: {success}")
        print(f"  幻觉引用: {has_halluc}")
        print(f"  已写入 {output_path}")

        _clear_gpu_memory()  # 清理GPU缓存，避免连续测试累积内存压力

    print(f"\n全部完成，共{len(queries)}条，日志文件: {output_path}")


if __name__ == "__main__":
    print("请在notebook中import本模块并调用run_batch_test()，同时传入已初始化的retrieval_pipeline和generation_pipeline")
