"""
scripts/11_generation_pipeline.py

医学生成流程整合pipeline：上下文组装 -> 证据评估 -> 草稿生成 -> 批判审查 -> 最终答案
包含引用幻觉校验：检测final_assembler改写阶段是否引入了不存在的source ID
"""

import re
import time
from typing import Dict, Any, List

from importlib import import_module

context_assembler_module = import_module("08_context_assembler")
prompt_templates_module = import_module("09_prompt_templates")
llm_generator_module = import_module("10_llm_generator")

ContextAssembler = context_assembler_module.ContextAssembler
PROMPT_STAGES = prompt_templates_module.PROMPT_STAGES
LLMGenerator = llm_generator_module.LLMGenerator


class MedicalGenerationPipeline:
    """
    流程：
      1. 上下文组装（ContextAssembler.assemble_context）
      2. 证据评估（evidence_evaluator，输出文字评估，作为下一步输入，不做chunk过滤）
      3. 草稿答案生成（answer_generator）
      4. 批判性审查（critical_reviewer，可选）
      5. 最终答案组装+翻译成中文（final_assembler）
      5.5 引用幻觉校验（_validate_citations）
      6. 后处理（引用来源列表、免责声明）
    """

    def __init__(
        self,
        llm_model_name: str = "deepseek-r1:7b",
        ollama_base_url: str = "http://localhost:11434",
        enable_evidence_evaluation: bool = True,
        enable_critical_review: bool = True,
        llm_timeout: int = 600,
    ):
        self.context_assembler = ContextAssembler()
        self.llm = LLMGenerator(
            model_name=llm_model_name, base_url=ollama_base_url, timeout=llm_timeout
        )
        self.enable_evidence_evaluation = enable_evidence_evaluation
        self.enable_critical_review = enable_critical_review

    def run(self, query: str, retrieved_docs: List[dict]) -> Dict[str, Any]:
        total_start = time.time()
        stage_times = {}
        stage_success = {}
        token_counts = {}

        # 1. 上下文组装
        t0 = time.time()
        context_result = self.context_assembler.assemble_context(retrieved_docs)
        stage_times["context_assembly"] = time.time() - t0
        stage_success["context_assembly"] = True

        selected_chunks = context_result["selected_chunks"]   # List[DocumentChunk]
        context_text = context_result["context_text"]

        evidence_evaluation_text = None
        draft_answer = None
        review_issues_text = None

        # 2. 证据评估（可选）—— 结果只作为文本传给下一步，不筛选chunk
        if self.enable_evidence_evaluation:
            t0 = time.time()
            stage_cfg = PROMPT_STAGES["evidence_evaluator"]
            prompt = stage_cfg.user_prompt_template.format(query=query, context=context_text)
            eval_result = self.llm.generate(
                prompt,
                system_prompt=stage_cfg.system_prompt,
                temperature=stage_cfg.temperature,
                max_tokens=stage_cfg.max_tokens,
            )
            stage_times["evidence_evaluator"] = time.time() - t0
            stage_success["evidence_evaluator"] = eval_result.success
            token_counts["evidence_evaluator"] = eval_result.completion_tokens_est

            if eval_result.success:
                evidence_evaluation_text = eval_result.text
            else:
                evidence_evaluation_text = "（证据评估生成失败，跳过此步骤）"

        # 3. 草稿答案生成
        t0 = time.time()
        stage_cfg = PROMPT_STAGES["answer_generator"]
        prompt = stage_cfg.user_prompt_template.format(
            query=query,
            context=context_text,
            evidence_evaluation=evidence_evaluation_text or "（未执行证据评估）",
        )
        draft_result = self.llm.generate(
            prompt,
            system_prompt=stage_cfg.system_prompt,
            temperature=stage_cfg.temperature,
            max_tokens=stage_cfg.max_tokens,
        )
        stage_times["answer_generator"] = time.time() - t0
        stage_success["answer_generator"] = draft_result.success
        token_counts["answer_generator"] = draft_result.completion_tokens_est

        if draft_result.success:
            draft_answer = draft_result.text
        else:
            return self._build_error_result(query, draft_result.error, total_start)

        # 4. 批判性审查（可选）+ 5. 最终答案
        final_answer = draft_answer
        if self.enable_critical_review:
            t0 = time.time()
            stage_cfg = PROMPT_STAGES["critical_reviewer"]
            prompt = stage_cfg.user_prompt_template.format(
                query=query, context=context_text, draft_answer=draft_answer
            )
            review_result = self.llm.generate(
                prompt,
                system_prompt=stage_cfg.system_prompt,
                temperature=stage_cfg.temperature,
                max_tokens=stage_cfg.max_tokens,
            )
            stage_times["critical_reviewer"] = time.time() - t0
            stage_success["critical_reviewer"] = review_result.success
            token_counts["critical_reviewer"] = review_result.completion_tokens_est

            if review_result.success:
                review_issues_text = review_result.text

                t0 = time.time()
                stage_cfg = PROMPT_STAGES["final_assembler"]
                prompt = stage_cfg.user_prompt_template.format(
                    query=query,
                    draft_answer=draft_answer,
                    review_issues=review_issues_text,
                )
                final_result = self.llm.generate(
                    prompt,
                    system_prompt=stage_cfg.system_prompt,
                    temperature=stage_cfg.temperature,
                    max_tokens=stage_cfg.max_tokens,
                )
                stage_times["final_assembler"] = time.time() - t0
                stage_success["final_assembler"] = final_result.success
                token_counts["final_assembler"] = final_result.completion_tokens_est

                if final_result.success:
                    final_answer = final_result.text
                # 失败则退回用draft_answer（英文）

        # 5.5 引用幻觉校验
        citation_check = self._validate_citations(final_answer, selected_chunks)

        # 6. 后处理
        final_answer = self._postprocess_answer(final_answer, selected_chunks)

        total_time = time.time() - total_start

        return {
            "query": query,
            "answer": final_answer,
            "context_metadata": context_result["metadata"],
            "generation_metrics": {
                "total_time_seconds": round(total_time, 2),
                "stage_times": {k: round(v, 2) for k, v in stage_times.items()},
                "token_counts": token_counts,
                "stage_success": stage_success,
            },
            "intermediate_results": {
                "evidence_evaluation": evidence_evaluation_text,
                "draft_answer": draft_answer,
                "review_issues": review_issues_text,
            },
            "citation_check": citation_check,
            "sources": self._format_sources(selected_chunks),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    def _validate_citations(self, answer: str, chunks: List) -> Dict[str, Any]:
        """
        检查答案里出现的[PMC\\d+]引用是否都在真实检索来源里，
        防止final_assembler改写阶段引入幻觉引用（编造/近似修改source ID）。
        """
        cited_ids = set(re.findall(r"\[?PMC\d+\]?", answer))
        cited_ids = {c.strip("[]") for c in cited_ids}

        real_ids = {c.source for c in chunks}

        hallucinated = cited_ids - real_ids

        check_result = {
            "cited_ids": sorted(cited_ids),
            "real_ids": sorted(real_ids),
            "hallucinated_ids": sorted(hallucinated),
            "has_hallucination": len(hallucinated) > 0,
        }

        if hallucinated:
            print(f"[警告] 检测到幻觉引用（真实来源中不存在）: {sorted(hallucinated)}")

        return check_result

    def _filter_chunks_by_evaluation(self, chunks: List[dict], evaluation: dict) -> List[dict]:
        relevant_ids = evaluation.get("relevant_doc_ids")
        if not relevant_ids:
            return chunks

        relevant_ids = set(str(i) for i in relevant_ids)
        filtered = [c for c in chunks if str(c.get("doc_id")) in relevant_ids]

        if not filtered:
            filtered = [
                c
                for c in chunks
                if any(
                    self._normalize_title(c.get("title", "")) in self._normalize_title(rid)
                    or self._normalize_title(rid) in self._normalize_title(c.get("title", ""))
                    for rid in relevant_ids
                )
            ]

        return filtered if filtered else chunks

    @staticmethod
    def _normalize_title(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    def _postprocess_answer(self, answer: str, chunks: list) -> str:
        answer = answer.strip()

        if "[来源" not in answer and "[source" not in answer.lower():
            sources = self._format_sources(chunks)
            if sources:
                refs = "\n\n**参考来源：**\n" + "\n".join(
                    f"[{i + 1}] {s['doc_id']} ({s.get('journal') or '未知期刊'}, {s.get('pub_date') or '未知日期'})"
                    for i, s in enumerate(sources)
                )
                answer += refs

        answer += (
            "\n\n---\n*本回答基于检索到的医学文献生成，仅供参考，不能替代专业医疗建议。"
            "具体诊疗请咨询执业医师。*"
        )
        return answer

    @staticmethod
    def _format_sources(chunks: list) -> List[dict]:
        return [
            {
                "doc_id": c.source,
                "chunk_id": c.chunk_id,
                "journal": c.metadata.get("journal"),
                "pub_date": c.metadata.get("pub_date"),
                "score": c.relevance_score,
            }
            for c in chunks
        ]

    @staticmethod
    def _build_error_result(query: str, error: str, total_start: float) -> dict:
        return {
            "query": query,
            "answer": None,
            "error": error,
            "generation_metrics": {"total_time_seconds": round(time.time() - total_start, 2)},
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
