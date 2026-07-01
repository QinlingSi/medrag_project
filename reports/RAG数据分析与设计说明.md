
# RAG数据分析与设计说明

## 1. 数据集事实

- **数据来源：** 来源于PubMed Central（PMC）开放获取子集oa_comm。JATS XML格式，使用Python的ElementTree库解析提取。
- **提取字段：** pmc_id、pmid、title、abstract、journal、pub_date。

### 字段完整性 （大小以及缺失情况）

- **原始记录数：** 3028篇（全部解析完成，无解析报错）
- **有效记录数：** 2775（abstract非空）

| 字段 | 缺失数 | 缺失率 |
|---|---|---|
| pmid | 275 | 9.08% |
| title | 0 | 0.00% |
| abstract | 253 | 8.36% |
| journal | 0 | 0.00% |
| pub_date | 12 | 0.40% |

**清洗决策**：abstract缺失率8.36%超过1%阈值，且abstract是RAG检索的核心内容，无法通过填充弥补，故直接丢弃这253篇记录，最终有效数据集为2775篇。

缺失原因为部分早期文献本身不含摘要，而非XML解析失败导致。

### 字段情况

#### 基础质量

- **超短摘要**（<30词）：200篇，占有效记录7.21%。经抽样核实大多为研究简报或快报类文章，内容本身简短，不存在质量问题，所以可以保留
- **编码错误**：0篇（0.00%）。PMC的XML文本编码质量良好，无需额外清洗。

#### 关键字段分析

- **journal**：共139个唯一值，可枚举，可以直接用作特定期刊文献的检索
- **pub_date**：年份跨度2003–2024，格式规范，可用年份进行过滤，且支持“近N年”的相关检索。但数据集中在2004和2005（占比高达98%），2024年仅23条，实际部署"近5年"过滤前建议补充更新批次数据。
- **pmid**：98.56%的记录存在pmid字段，格式规范（纯数字），可直接拼接为 https://pubmed.ncbi.nlm.nih.gov/{pmid}/ 作为原文溯源链接。


## 2. 文本长度分布

使用embedding模型 **BAAI/bge-base-en-v1.5** 的真实tokenizer，
对"标题+摘要"拼接文本统计token数（共2775篇）：

| 分位数 | Token数 |
|---|---|
| P5 | 44 |
| P25 | 261 |
| P50 | 352 |
| P75 | 440 |
| P90 | 527 |
| P95 | 576 |
| P99 | 686 |
| 最大值 | 952 |

- 均值：343 tokens
- **P95（576 tokens）已超过bge-base-en-v1.5的512 token上限**
- 超过512 tokens的文章占比：**11.82%（328/2775篇）**
- 超限比例较高，不属于"少数长尾"，因此需对全部文章制定分割策略

## 3. 领域语言特性

### 高频术语

通过高频词统计（去停用词后Top20）与分层抽样人工阅读，本数据集高频实词集中在：cells、expression、gene、genes、protein等与生物学基因学等基础研究为主的相关内容。而期刊分布以PLoS Biology、BMC Bioinformatics、BMC Genomics等为主，也印证了上述判断。

### 结构特点

通过分层抽样（短/中/长各抽8篇，按四分位数（P25=261、P75=440 tokens）分界）：

- **长文本（≥440 tokens）**：100%严格遵循 Background→Methods→Results→Conclusions 四段式IMRaD结构
- **中等文本（261–440 tokens）**：同样具备完整四段结，小节标签词存在变体（如Design/Discussion代替Methods/Conclusions）
- **短文本（<261 tokens）**：结构分化明显，部分为简化版三段式，部分为单段连续叙述

**结论**：IMRaD结构完整度与文本长度强正相关；短文本中可能混有非研究类内容，检索时需注意。

### 术语与表述风格

- 缩写密度中等偏高，普遍遵循"首次出现给全称+括号缩写"的写法
  （如"hepatitis B virus (HBV)"）
- 全部为学术写作风格，无口语化表述

## 4. 分割策略及原因

### 策略选择

首先对三种策略逐一判断：

- **整体不分割**：要求P95 < 400 tokens，本数据集P95=576，不满足，排除
- **按语义章节分割**：适合结构极其清晰的数据，本数据集短文本中存在大量无IMRaD结构的文章，无法统一按章节标题分割，排除
- **重叠滑动窗口** ：本数据集11.82%文章超过512 tokens， 存在明显长尾，采用此策略

### 实现参数

- 工具：LangChain `RecursiveCharacterTextSplitter`
- `chunk_size = 450 tokens`
- `chunk_overlap = 80 tokens`
- 长度计算：使用bge-base-en-v1.5真实tokenizer
- 分隔符优先级：段落 > 句子 > 空格

### 验证结果

- 总计生成：3768个chunk（原始2775篇）
- 76.1%的文章未被分割（整篇为单一chunk）
- 分块后超过512 tokens的chunk数：**0**


## 5. 其他补充说明

- **pub_date年份分布高度集中在2004–2005年**，与数据来源批次（oa_comm特定PMC ID区间）有关，不代表PubMed整体年份分布，建议后续补充更新批次数据。
- **工具链**：数据提取（Python + ElementTree）→ 探索性分析（Jupyter Notebook + transformers）→分块实现（LangChain RecursiveCharacterTextSplitter）
- - **代码结构**：
  - `notebooks/01_data_analysis.ipynb`：步骤1–5探索性分析
  - `scripts/02_chunking.py`：分块逻辑
  - `data/processed/dataset.json`：清洗后的数据（2775篇）
  - `data/processed/chunks.json`：分块结果（3768个chunk）
    












  