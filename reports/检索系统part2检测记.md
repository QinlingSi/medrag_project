# 检索系统 Part 2 测试记录

测试日期：2026-07-31
测试环境：本地 Mac Apple Silicon, 8GB 统一内存, conda env `medrag`
测试 query：`What is the effect of metformin on cardiovascular disease?`（英文，中文query问题见文末说明）

---

## 1. 向量检索（MultiPathRetriever.vector_search）

模型：`BAAI/bge-small-en-v1.5`，通过 `FlagModel.encode_queries()` 编码 query，`collection.query(query_embeddings=...)` 检索

```
1 PMC2644685    0.794  Metformin treatment in diabetes and heart failure...
2 PMC2991324    0.791  Long-term effect of metformin on blood glucose con...
3 PMC2946277_chunk0  0.789  Effects of oral glucose-lowering drugs on long ter...
4 PMC2705820    0.776  Antihypertensive therapy, new-onset diabetes, and...
5 PMC2723076_chunk0  0.764  A cardiologic approach to non-insulin antidiabetic...
```

相似度范围 0.76–0.79，结果均与 metformin / 心血管相关，判定正常。

**修复的问题：**
- ChromaDB 用 `query_texts=` 会默认调用其内置 embedding 模型（非 bge-small），导致向量空间不匹配 → 改为手动编码 + `query_embeddings=`
- `query_info['vector_query']` 已带 BGE 指令前缀，若再传入 `encode_queries()` 会被二次加前缀 → 改用 `cleaned_query`
- 中文 query 用纯英文模型编码后检索结果完全不相关（已验证，见文末）

---

## 2. BM25 关键词检索（MultiPathRetriever.keyword_search）

分词：jieba，索引：`rank_bm25.BM25Okapi`，语料来自 `chunks.parquet` 全量 170,237 条

```
1 PMC2694802         24.067  Does metformin affect ovarian morphology in patien...
2 PMC2847989         22.549  Assessment of efficacy and tolerability of once-da...
3 PMC2807458         22.235  Metformin Induces a Dietary Restriction–Like State...
4 PMC2613390_chunk0  21.819  Cell cycle arrest in Metformin treated breast canc...
5 PMC1974811_chunk0  21.81   Rosiglitazone RECORD study: glucose control outcom...
```

中英文 query 均可正常检索。

---

## 3. 融合策略对比（MultiPathRetriever.fusion_search）

### simple（去重合并，按原始 rank 排序）
```
1 PMC2644685          0.7944  Metformin treatment in diabetes and heart failure:
2 PMC2991324          0.7908  Long-term effect of metformin on blood glucose con
3 PMC2946277_chunk0   0.7893  Effects of oral glucose-lowering drugs on long ter
4 PMC2705820          0.7759  Antihypertensive therapy, new-onset diabetes, and
5 PMC2723076_chunk0   0.7641  A cardiologic approach to non-insulin antidiabetic
```

### rrf（Reciprocal Rank Fusion, k=60）
```
1 PMC2644685      0.0164  Metformin treatment in diabetes and heart failure:
2 PMC2875210      0.0164  Using family history information to promote health
3 PMC2991324      0.0161  Long-term effect of metformin on blood glucose con
4 PMC2729049      0.0161  Peroxisome Proliferator-Activated Receptor Agonist
5 PMC2946277_chunk0  0.0159  Effects of oral glucose-lowering drugs on long ter
```

### weighted（向量0.6 + BM25 0.4，minmax归一化）
```
1 PMC2644685      0.6     Metformin treatment in diabetes and heart failure:
2 PMC2991324      0.5377  Long-term effect of metformin on blood glucose con
3 PMC2946277_chunk0  0.5122  Effects of oral glucose-lowering drugs on long ter
4 PMC2875210      0.4     Using family history information to promote health
5 PMC2705820      0.2801  Antihypertensive therapy, new-onset diabetes, and
```

**观察**：当前测试 query 下三种策略 top 结果高度重合（向量检索本身已足够强），BM25 增量差异不明显。建议后续用更依赖关键词精确匹配的 query 做对比测试，观察策略间差异。

**修复的问题**：`_fuse_simple` 最初未重新排序（只是去重合并），导致输出顺序等价于随机遍历顺序；已修复为按原始 rank 排序并重新赋值 rank。

---

## 4. Reranker 基础验证（MultiCriteriaReranker.rerank）

模型：`BAAI/bge-reranker-base`（本地缓存 `bge-reranker-base-cache/`）
权重：relevance 0.6 / recency 0.25 / authority 0.15

```
测试 candidates:
  test1: "Metformin treatment in diabetes and heart failure patients."
  test2: "The Soul's Wisdom: Stories of Living and Dying."

1  test1  final_score=0.641  relevance_score=0.735
2  test2  final_score=0.2001 relevance_score=0.0001
```

相关 / 不相关内容区分度非常明显（relevance 0.735 vs 0.0001）。

**修复的问题**：`journal`/`pub_date` 字段在 parquet 中缺失值为 pandas NaN（非 None/空字符串），原 `_compute_authority`/`_compute_recency` 未处理该类型，导致 `TypeError: unsupported operand type(s) for *: 'float' and 'NoneType'`；已加 NaN/None/非字符串类型判断。

---

## 5. 完整 Pipeline 端到端测试（MedRAGPipeline.run）

query → 04查询处理 → 05多路检索融合(rrf) → 补充metadata → 06重排序，top_k_final=10

```
1  PMC2946277_chunk0  0.3029  Effects of oral glucose-lowering drugs on long term outcomes
2  PMC2546413_chunk0  0.2297  Fatal hemolytic anemia associated with metformin: A case rep
3  PMC2566605_chunk0  0.2016  Effect of Adjunct Metformin Treatment in Patients with Type-
4  PMC2991324         0.1554  Long-term effect of metformin on blood glucose control in no
5  PMC1974811_chunk0  0.1277  Rosiglitazone RECORD study: glucose control outcomes at 18 m
6  PMC2664796_chunk0  0.1276  Diabetic cardiomyopathy: effects of fenofibrate and metformi
7  PMC2940872         0.1254  Therapies for type 2 diabetes: lowering HbA1c and associated
8  PMC2797799         0.124   Lifestyle modification and metformin as long-term treatment
9  PMC2906460_chunk1  0.121   Conversely, pioglitazone had no impact on fasting insulin, t
10 PMC2169248_chunk0  0.1201  Metformin-induced lactic acidosis: a case series. Introducti
```

全部结果与 metformin 相关主题（心血管/代谢/长期疗效/不良反应）相符，分数呈合理梯度递减。**Part 2 端到端跑通。**

---

## 6. 已知限制 / 待办

- **中文 query 向量检索问题**：`bge-small-en-v1.5` 为纯英文模型，中文 query 编码后向量空间与英文文献库不匹配，实测检索结果完全不相关（相似度仅 0.14–0.16，内容不相关）；换成对应英文 query 后恢复正常（相似度 0.76–0.79）。已评估三个方案（关键词拼接 / 本地Ollama整句翻译 / 换双语embedding模型），倾向方案二（翻译），已与 Daniel 对齐先跑通英文版本，中文方案留待后续实现。
- **期刊权威度字典**：`JOURNAL_AUTHORITY_WEIGHTS` 目前是占位权重（几个知名期刊硬编码 + 默认0.5），未接入真实 JCR/SJR 影响因子数据源，属已知 TODO。
- **融合策略选择**：三种策略在当前测试 query 下区分度不大，需要更多样化的测试 query 验证差异，以确定生产环境默认策略。

---

## 交付文件

- `scripts/05_multipath_retrieval.py`
- `scripts/06_reranker.py`
- `scripts/07_retrieval_pipeline.py`
- `notebooks/04_retrieval_dev.ipynb`

已提交 GitHub commit `c885474`。
