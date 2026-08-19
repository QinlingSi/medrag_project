import hashlib
import json
import time
from collections import OrderedDict


class CacheManager:
    def __init__(self, max_size=500, ttl_seconds=24 * 3600, temp_threshold=0.3):
        self.store = OrderedDict()
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.temp_threshold = temp_threshold

    def _make_key(self, query: str, context: str) -> str:
        raw = json.dumps({"q": query, "c": context}, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, query: str, context: str):
        key = self._make_key(query, context)
        if key not in self.store:
            return None
        value, ts = self.store[key]
        if time.time() - ts > self.ttl_seconds:
            del self.store[key]
            return None
        self.store.move_to_end(key)  # 刚被用过，挪到末尾（末尾=最近使用）
        return value

    def set(self, query: str, context: str, value, temperature: float = 0.0):
        if temperature > self.temp_threshold:
            return  # 温度太高，不缓存
        key = self._make_key(query, context)
        self.store[key] = (value, time.time())
        self.store.move_to_end(key)
        if len(self.store) > self.max_size:
            self.store.popitem(last=False)  # 淘汰最前面的，也就是最久没用的


if __name__ == "__main__":
    # 测试温度限制
    cache = CacheManager(max_size=3, ttl_seconds=3600, temp_threshold=0.3)

    cache.set("q1", "", "answer1", temperature=0.1)  # 低温度，应该缓存成功
    cache.set("q2", "", "answer2", temperature=0.8)  # 高温度，应该被拒绝

    print("q1(低温度存入)get结果:", cache.get("q1", ""))
    print("q2(高温度，应该没存进去)get结果:", cache.get("q2", ""))

    # 测试LRU淘汰：max_size=3，存4条，最早且没被访问过的那条应该被挤掉
    cache2 = CacheManager(max_size=3, ttl_seconds=3600, temp_threshold=0.3)
    cache2.set("a", "", "answer_a", temperature=0.1)
    cache2.set("b", "", "answer_b", temperature=0.1)
    cache2.set("c", "", "answer_c", temperature=0.1)
    cache2.set("d", "", "answer_d", temperature=0.1)  # 第4条进来，容量超了

    print("a(应该被淘汰，因为最早存入且没再被访问):", cache2.get("a", ""))
    print("d(最新存入，应该还在):", cache2.get("d", ""))