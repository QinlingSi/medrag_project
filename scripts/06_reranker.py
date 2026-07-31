# scripts/06_reranker.py
import os
os.environ["HF_HUB_OFFLINE"] = "1"

import re
from datetime import datetime
from pandas import isna as pd_isna

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

RERANKER_MODEL = "../bge-reranker-base-cache"

DEFAULT_CRITERIA_WEIGHTS = {
    "relevance": 0.6,
    "recency": 0.25,
    "authority": 0.15,
}

# TODO: 目前是占位权重，后续考虑接入JCR影响因子/SJR等真实数据源替换
JOURNAL_AUTHORITY_WEIGHTS = {
    "nature": 1.0,
    "science": 1.0,
    "the lancet": 1.0,
    "new england journal of medicine": 1.0,
    "jama": 0.9,
    "bmj": 0.9,
    "plos one": 0.6,
}
DEFAULT_JOURNAL_WEIGHT = 0.5
RECENCY_WINDOW_YEARS = 15


class MultiCriteriaReranker:
    def __init__(self, model_name=RERANKER_MODEL, criteria_weights=None, device=None):
        print("加载reranker模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        self.weights = criteria_weights or DEFAULT_CRITERIA_WEIGHTS
        assert abs(sum(self.weights.values()) - 1.0) < 1e-6, "criteria_weights权重之和必须为1"

    def rerank(self, query, candidates, top_k=10):
        """
        query: 原始查询文本（建议用英文cleaned_query，跟reranker模型语言一致）
        candidates: MultiPathRetriever融合后的结果列表，每条需要有 text/chunk_id，
                    metadata里最好带 pub_date 和 journal（没有的话recency/authority按默认分处理）
        """
        relevance_scores = self._compute_relevance(query, candidates)

        scored = []
        for cand, rel_score in zip(candidates, relevance_scores):
            recency_score = self._compute_recency(cand.get("pub_date"))
            authority_score = self._compute_authority(cand.get("journal"))

            final_score = (
                self.weights["relevance"] * rel_score
                + self.weights["recency"] * recency_score
                + self.weights["authority"] * authority_score
            )
            item = dict(cand)
            item["relevance_score"] = round(rel_score, 4)
            item["recency_score"] = round(recency_score, 4)
            item["authority_score"] = round(authority_score, 4)
            item["final_score"] = round(final_score, 4)
            scored.append(item)

        scored.sort(key=lambda x: x["final_score"], reverse=True)
        for rank, item in enumerate(scored):
            item["rank"] = rank + 1
        return scored[:top_k]

    def _compute_relevance(self, query, candidates):
        pairs = [[query, cand["text"]] for cand in candidates]
        with torch.no_grad():
            inputs = self.tokenizer(
                pairs, padding=True, truncation=True, max_length=512, return_tensors="pt"
            ).to(self.device)
            logits = self.model(**inputs).logits.view(-1)
            scores = torch.sigmoid(logits).cpu().tolist()
        return scores

    def _compute_recency(self, pub_date):
            if pub_date is None or (isinstance(pub_date, float) and pd_isna(pub_date)):
                return 0.5
            year = self._extract_year(pub_date)
            if year is None:
                return 0.5
            current_year = datetime.now().year
            age = current_year - year
            score = 1 - age / RECENCY_WINDOW_YEARS
            return max(0.0, min(1.0, score))

    def _extract_year(self, pub_date):
        match = re.search(r"(19|20)\d{2}", str(pub_date))
        return int(match.group()) if match else None

    def _compute_authority(self, journal):
        if journal is None or (isinstance(journal, float) and pd_isna(journal)):
            return DEFAULT_JOURNAL_WEIGHT
        if not isinstance(journal, str) or not journal.strip():
            return DEFAULT_JOURNAL_WEIGHT
        key = journal.strip().lower()
        return JOURNAL_AUTHORITY_WEIGHTS.get(key, DEFAULT_JOURNAL_WEIGHT)