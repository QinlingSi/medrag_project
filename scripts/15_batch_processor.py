import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


def batch_generate(queries: list, generate_fn, max_workers=None) -> list:
    if max_workers is None:
        max_workers = min(len(queries), os.cpu_count() or 4)

    results = [None] * len(queries)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(generate_fn, q): i for i, q in enumerate(queries)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                results[idx] = {"error": str(e)}
    return results


def fake_generate(query: str) -> str:
    time.sleep(1)
    if query == "query3":
        raise ValueError("模拟生成失败，比如模型超时或者返回格式错误")
    return f"answer for: {query}"


if __name__ == "__main__":
    queries = ["query1", "query2", "query3", "query4", "query5"]

    results = batch_generate(queries, fake_generate)

    print("结果(query3应该是error，其他4条应该正常):")
    for q, r in zip(queries, results):
        print(f"  {q} -> {r}")

    print(f"\n当前机器CPU核心数: {os.cpu_count()}，没手动指定max_workers时会自动用这个数(不超过query条数)")