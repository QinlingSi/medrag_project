"""
scripts/08_context_assembler.py
上下文组装器 —— 将检索+重排后的文档块组装成喂给LLM的context
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import re

try:
    from transformers import AutoTokenizer
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False


@dataclass
class DocumentChunk:
    text: str
    metadata: Dict[str, Any]
    relevance_score: float
    source: str         
    chunk_id: str


class ContextAssembler:
    def __init__(
        self,
        tokenizer_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        max_context_tokens: int = 3000,
    ):
        self.max_context_tokens = max_context_tokens
        self.tokenizer = None
        self._use_fallback_estimate = False

        if not _TRANSFORMERS_AVAILABLE:
            print("[警告] transformers 未安装，token估算将降级为 字符数/4 近似法")
            self._use_fallback_estimate = True
            return

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            print(f"[ContextAssembler] tokenizer 加载成功: {tokenizer_name}")
        except Exception as e:
            print(f"[警告] tokenizer 加载失败 ({e})，降级为 字符数/4 近似法")
            self._use_fallback_estimate = True

    def estimate_tokens(self, text: str) -> int:
        """估算文本的token数量"""
        if self._use_fallback_estimate or self.tokenizer is None:
            return max(1, len(text) // 4)
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _convert_to_chunks(self, retrieved_docs: List[Dict[str, Any]]) -> List["DocumentChunk"]:
        """将pipeline输出的dict列表转换为DocumentChunk对象列表"""
        chunks = []
        for doc in retrieved_docs:
            chunk_id = doc.get("chunk_id", "")
            doc_id = chunk_id.split("_chunk")[0] if "_chunk" in chunk_id else chunk_id

            chunks.append(DocumentChunk(
                text=doc.get("text", ""),
                metadata={
                    "pub_date": doc.get("pub_date"),
                    "journal": doc.get("journal"),
                    "rank": doc.get("rank"),
                    "fusion_source": doc.get("source"),
                    "relevance_score": doc.get("relevance_score"),
                    "recency_score": doc.get("recency_score"),
                    "authority_score": doc.get("authority_score"),
                },
                relevance_score=doc.get("final_score", doc.get("score", 0.0)),
                source=doc_id,
                chunk_id=chunk_id,
            ))
        return chunks

    
    def _jaccard_similarity(self, text_a: str, text_b: str) -> float:
        """计算两段文本的Jaccard相似度（基于词集合）"""
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union)

    def _deduplicate(
        self,
        chunks: List["DocumentChunk"],
        similarity_threshold: float = 0.8,
    ) -> List["DocumentChunk"]:
        """基于Jaccard相似度去重，优先保留relevance_score更高的chunk"""
        sorted_chunks = sorted(chunks, key=lambda c: c.relevance_score, reverse=True)

        unique_chunks: List["DocumentChunk"] = []
        for candidate in sorted_chunks:
            is_duplicate = False
            for kept in unique_chunks:
                sim = self._jaccard_similarity(candidate.text, kept.text)
                if sim >= similarity_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_chunks.append(candidate)

        return unique_chunks

    def _diversify_and_rank(
        self,
        chunks: List["DocumentChunk"],
        diversity_penalty: float = 0.7,
    ) -> List["DocumentChunk"]:
        """按相关性排序，并对同一来源(source/doc_id)的重复出现进行降权，保证多样性"""
        sorted_chunks = sorted(chunks, key=lambda c: c.relevance_score, reverse=True)

        source_seen_count: Dict[str, int] = {}
        adjusted = []
        for chunk in sorted_chunks:
            seen = source_seen_count.get(chunk.source, 0)
            penalty = diversity_penalty ** seen  # 第1次出现 seen=0，penalty=1，不受影响
            adjusted_score = chunk.relevance_score * penalty
            adjusted.append((adjusted_score, chunk))
            source_seen_count[chunk.source] = seen + 1

        # 按调整后的分数重新排序
        adjusted.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in adjusted]

    def _truncate_at_sentence(self, text: str, max_tokens: int) -> str:
        """将文本截断到max_tokens以内，并尽量在完整句子处结束"""
        if self.estimate_tokens(text) <= max_tokens:
            return text

        # 先按字符做粗略二分，找到大致不超过max_tokens的长度
        low, high = 0, len(text)
        while low < high:
            mid = (low + high + 1) // 2
            if self.estimate_tokens(text[:mid]) <= max_tokens:
                low = mid
            else:
                high = mid - 1
        truncated = text[:low]

        window_size = max(400, int(len(truncated) * 0.3))
        search_start = max(0, len(truncated) - window_size)

        window = truncated[search_start:]
        sentence_enders = ["。", ".", "！", "!", "？", "?"]
        last_pos = -1
        for ender in sentence_enders:
            pos = window.rfind(ender)
            if pos > last_pos:
                last_pos = pos

        if last_pos != -1:
            return truncated[:search_start + last_pos + 1]
        # 找不到句子边界，退而求其次直接截断
        return truncated

    def _analyze_sources(self, chunks: List["DocumentChunk"]) -> Dict[str, Any]:
        """统计选中chunk的来源分布"""
        source_counts: Dict[str, int] = {}
        for c in chunks:
            source_counts[c.source] = source_counts.get(c.source, 0) + 1
        return {
            "unique_sources": len(source_counts),
            "source_distribution": source_counts,
        }

    def assemble_context(self, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """主入口：将检索结果组装成最终喂给LLM的context"""
        all_chunks = self._convert_to_chunks(retrieved_docs)

        unique_chunks = self._deduplicate(all_chunks)

        ranked_chunks = self._diversify_and_rank(unique_chunks)

        selected_chunks: List["DocumentChunk"] = []
        context_parts: List[str] = []
        current_tokens = 0

        for chunk in ranked_chunks:
            header = f"[来源: {chunk.source} | 期刊: {chunk.metadata.get('journal', '未知')} | 发表: {chunk.metadata.get('pub_date', '未知')}]\n"
            piece = header + chunk.text + "\n\n"
            piece_tokens = self.estimate_tokens(piece)

            if current_tokens + piece_tokens <= self.max_context_tokens:
                context_parts.append(piece)
                current_tokens += piece_tokens
                selected_chunks.append(chunk)
            else:
                remaining_budget = self.max_context_tokens - current_tokens
                # 剩余空间太小（比如不足50 token），直接放弃这一条，不做无意义的截断
                if remaining_budget > 50:
                    truncated_text = self._truncate_at_sentence(chunk.text, remaining_budget - self.estimate_tokens(header))
                    if truncated_text.strip():
                        piece = header + truncated_text + "\n\n"
                        context_parts.append(piece)
                        current_tokens += self.estimate_tokens(piece)
                        selected_chunks.append(chunk)
                break  

        final_context = "".join(context_parts)

        context_metadata = {
            "total_chunks_retrieved": len(retrieved_docs),
            "unique_chunks_after_dedup": len(unique_chunks),
            "chunks_selected": len(selected_chunks),
            "estimated_tokens": self.estimate_tokens(final_context),
            "chunk_sources": self._analyze_sources(selected_chunks),
        }

        return {
            "context_text": final_context,
            "metadata": context_metadata,
            "selected_chunks": selected_chunks,
        }
