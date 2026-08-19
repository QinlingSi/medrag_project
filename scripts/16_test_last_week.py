import sys, os, json, time
from importlib import import_module

sys.path.append(os.path.dirname(__file__))

retrieval_module = import_module("07_retrieval_pipeline")
generation_module = import_module("11_generation_pipeline")
test_module = import_module("12_test_generation_pipeline")
eval_module = import_module("13_answer_evaluator")
cache_module = import_module("14_cache_manager")
batch_module = import_module("15_batch_processor")

AnswerEvaluator = eval_module.AnswerEvaluator
CacheManager = cache_module.CacheManager
batch_generate = batch_module.batch_generate

TEST_QUERIES = test_module.TEST_QUERIES

GROUND_TRUTHS = {
    "What is the effect of metformin on cardiovascular disease?":
        "Metformin has a neutral to beneficial effect on cardiovascular outcomes in type 2 diabetes patients. "
        "The UKPDS trial showed reduced risk of myocardial infarction and diabetes-related death in overweight "
        "patients treated with metformin. It is generally considered cardioprotective rather than harmful.",

    "What are the differences between metformin and sulfonylureas in treating type 2 diabetes?":
        "Metformin lowers blood glucose mainly by reducing hepatic glucose production and improving insulin "
        "sensitivity, without stimulating insulin secretion, so it carries low risk of hypoglycemia and is "
        "weight-neutral. Sulfonylureas stimulate pancreatic beta cells to secrete more insulin, which carries "
        "higher risk of hypoglycemia and is often associated with weight gain.",

    "What are the common side effects of metformin?":
        "The most common side effects of metformin are gastrointestinal: diarrhea, nausea, vomiting, abdominal "
        "discomfort, and metallic taste, often dose-dependent. Long-term use can cause vitamin B12 deficiency. "
        "A rare but serious risk is lactic acidosis, particularly in patients with renal impairment.",

    "What is the recommended dosage of metformin for elderly patients?":
        "For elderly patients, metformin is typically started at a low dose such as 500mg once or twice daily "
        "and titrated gradually based on tolerance and renal function, with a maximum daily dose often around "
        "2000mg. Renal function (eGFR) should be checked before and during treatment; metformin is generally "
        "avoided when eGFR is below 30. It does not need to be combined with insulin.",

    "Can metformin be used together with insulin?":
        "Yes, metformin can be safely combined with insulin in the treatment of type 2 diabetes. This combination "
        "can reduce the total insulin dose needed, help limit insulin-associated weight gain, and improve overall "
        "glycemic control compared to insulin alone.",
}

# ---- 初始化，和06_test.ipynb里一样的参数 ----
retrieval_pipeline = retrieval_module.MedRAGPipeline(
    chunks_path='../data/processed/chunks.parquet',
    chroma_db_path='../data/processed/chroma_db'
)

generation_pipeline = generation_module.MedicalGenerationPipeline(
    llm_model_name="deepseek-r1:7b",
    ollama_base_url="http://localhost:11434",
    enable_evidence_evaluation=False,
    enable_critical_review=False,
    llm_timeout=600
)

cache = CacheManager(max_size=100, ttl_seconds=24 * 3600, temp_threshold=0.3)
evaluator = AnswerEvaluator()


def _context_key(retrieved_docs):
    # 用检索出来的文档id拼一个字符串当context，具体字段名可能要按你实际retrieved_docs结构调整
    try:
        return "|".join(str(d.get("id") or d.get("pmc_id") or d.get("doc_id")) for d in retrieved_docs)
    except Exception:
        return str(retrieved_docs)[:500]

def generate_with_cache(query: str) -> dict:
    print(f"[开始处理] {query}")
    retrieved_docs = retrieval_pipeline.run(query, top_k_final=5)
    print(f"[检索完成] {query}")
    ctx_key = _context_key(retrieved_docs)

    cached = cache.get(query, ctx_key)
    if cached is not None:
        print(f"[缓存命中] {query}")
        return {"answer": cached, "from_cache": True}

    result = generation_pipeline.run(query, retrieved_docs)
    print(f"[生成完成] {query}")
    answer = result.get("answer") or ""
    cache.set(query, ctx_key, answer, temperature=0.3)
    return {"answer": answer, "from_cache": False, "raw_result": result}


if __name__ == "__main__":
    print("===== 第一轮：正常生成（预期全部缓存未命中）=====")
    start = time.time()
    results_1 = batch_generate(TEST_QUERIES, generate_with_cache, max_workers=1)
    print(f"第一轮耗时: {time.time() - start:.1f}秒\n")

    print("===== 第二轮：重复跑同样的query（预期全部缓存命中，速度应该明显变快）=====")
    start = time.time()
    results_2 = batch_generate(TEST_QUERIES, generate_with_cache, max_workers=1)
    print(f"第二轮耗时: {time.time() - start:.1f}秒\n")

    print("===== 评估结果 =====")
    output_path = "../reports/eval_cache_batch_test_log.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for query, r in zip(TEST_QUERIES, results_1):
            gt = GROUND_TRUTHS.get(query, "")
            eval_result = evaluator.evaluate(r["answer"], gt)
            log_entry = {"query": query, "answer": r["answer"], "eval": eval_result}
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            print(f"\nQ: {query}")
            print(f"  相似度rouge-l f1: {eval_result['similarity']['rouge-l']['f']:.3f}")
            print(f"  关键信息召回率: {eval_result['key_info_recall']:.3f}")
            print(f"  幻觉风险: {eval_result['hallucination_risk']:.3f}")
    print(f"\n完整结果写入: {output_path}")