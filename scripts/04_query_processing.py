"""
04_query_processing.py
检索系统 - 查询理解与增强模块

输入: 用户自然语言医学查询
输出: 结构化的增强查询信息(清洗后的query、识别的实体、扩展的同义词、
      向量检索query、关键词检索query、过滤条件)
"""

import re
import unicodedata


# ==================== 配置数据 ====================

MEDICAL_SYNONYMS = {
    "mi": ["myocardial infarction", "heart attack"],
    "二甲双胍": ["metformin"],
    "阿司匹林": ["aspirin"],
    "华法林": ["warfarin"],
    "胰岛素": ["insulin"],
    # TODO: 后续从UMLS/MeSH等标准术语库扩充更全面的中英对照
}

MEDICAL_PATTERNS = {
    'drug': r'(?<![A-Za-z0-9])(aspirin|metformin|atorvastatin|warfarin|insulin)(?![A-Za-z0-9])|(二甲双胍|阿司匹林|华法林|胰岛素)',
}



# ==================== 处理函数 ====================

def clean_query(query: str) -> str:
    """
    对用户输入的医学查询做基础清洗:
    1. 全角转半角(处理中文输入法产生的全角标点/数字)
    2. 首尾空格去除 + 中间多余空白压缩为单个空格
    3. 保留原始大小写(大小写敏感的匹配交给后续同义词/实体识别阶段处理)
    """
    if not query or not query.strip():
        return ""

    # 全角转半角 (NFKC规范化会把全角字符转成对应的半角形式)
    query = unicodedata.normalize('NFKC', query)

    # 压缩空白: 多个空格/制表符/换行 -> 单个空格, 并去除首尾空格
    query = re.sub(r'\s+', ' ', query).strip()

    return query


def extract_medical_entities(query: str) -> dict:
    """
    从query中识别医学实体(目前仅支持drug类别，支持中英文药名)。
    返回格式: {"drug": ["aspirin", "二甲双胍"], ...}
    """
    entities = {}

    for entity_type, pattern in MEDICAL_PATTERNS.items():
        raw_matches = re.findall(pattern, query, flags=re.IGNORECASE)
        matches = []
        for m in raw_matches:
            if isinstance(m, tuple):
                matches.extend([g for g in m if g])
            else:
                matches.append(m)

        if matches:
            seen = []
            for m in matches:
                m_norm = m.lower()
                if m_norm not in seen:
                    seen.append(m_norm)
            entities[entity_type] = seen

    return entities


def expand_synonyms(query: str, entities: dict) -> list:
    """
    对query做同义词扩展，返回额外应该被加入检索的同义词列表(不含原词)。

    扩展来源有两个:
    1. 已识别的实体(entities) —— 拿实体值去MEDICAL_SYNONYMS查
    2. query原文中可能出现的缩写词(如"MI") —— 因为缩写不一定会被MEDICAL_PATTERNS
       识别为实体(目前MEDICAL_PATTERNS只覆盖drug类)，需要单独按词匹配

    返回格式: 去重后的同义词列表，如 ["myocardial infarction", "heart attack"]
    """
    expanded = []

    # 1. 从已识别实体里查同义词
    for entity_list in entities.values():
        for entity in entity_list:
            if entity in MEDICAL_SYNONYMS:
                expanded.extend(MEDICAL_SYNONYMS[entity])

    # 2. 单独扫描query中的词，匹配缩写词典的key
    #    按词边界切分(排除中文场景下\b失效的问题，用非字母数字做分隔)
    query_words = re.findall(r'[A-Za-z]+', query.lower())
    for word in query_words:
        if word in MEDICAL_SYNONYMS:
            expanded.extend(MEDICAL_SYNONYMS[word])

    # 去重，保持顺序
    seen = []
    for term in expanded:
        if term not in seen:
            seen.append(term)

    return seen

def prepare_vector_query(query: str) -> str:
    """
    生成用于向量检索的query版本。
    按BGE模型的最佳实践，加上指令前缀，能提升向量检索的相关性。
    """
    return f"Represent this question for searching relevant passages: {query}"


def prepare_keyword_query(query: str, expanded_terms: list) -> str:
    """
    生成用于关键词检索(如BM25)的query版本。
    关键词检索不需要指令前缀，但可以把同义词扩展词一并纳入，
    增加关键词匹配到相关文档的概率。
    """
    if expanded_terms:
        return query + " " + " ".join(expanded_terms)
    return query


def extract_filters(query: str) -> dict:
    """
    从query中提取可结构化的过滤条件，目前支持:
    - year_from / year_to: 时间范围过滤

    识别的表达方式:
    - "近N年" / "最近N年" -> year_from = 当前年 - N
    - "YYYY年以后" / "since YYYY" / "after YYYY" -> year_from = YYYY
    - "YYYY年以前" / "before YYYY" -> year_to = YYYY

    返回格式: {"year_from": 2021, "year_to": None} (未命中的字段为None，不加入结果时省略)
    """
    from datetime import datetime
    filters = {}
    current_year = datetime.now().year

    # 中文: 近N年 / 最近N年
    m = re.search(r'(?:最近|近)\s*(\d+)\s*年', query)
    if m:
        n = int(m.group(1))
        filters['year_from'] = current_year - n

    # 英文/中文: YYYY年以后 / since YYYY / after YYYY
    m = re.search(r'(\d{4})\s*年?\s*(?:以后|之后)|(?:since|after)\s*(\d{4})', query, flags=re.IGNORECASE)
    if m:
        year = m.group(1) or m.group(2)
        filters['year_from'] = int(year)

    # 英文/中文: YYYY年以前 / before YYYY
    m = re.search(r'(\d{4})\s*年?\s*(?:以前|之前)|before\s*(\d{4})', query, flags=re.IGNORECASE)
    if m:
        year = m.group(1) or m.group(2)
        filters['year_to'] = int(year)

    return filters

def process_medical_query(query: str) -> dict:
    """
    查询理解与增强模块的统一入口。
    输入原始用户query，输出结构化的增强查询信息，供后续检索模块使用。

    返回格式:
    {
        "original_query": 原始输入,
        "cleaned_query": 清洗后的query,
        "entities": {"drug": [...]},
        "expanded_terms": [...],
        "vector_query": "加了BGE指令前缀的向量检索版本",
        "keyword_query": "拼接同义词的关键词检索版本",
        "filters": {"year_from": ..., "year_to": ...}
    }
    """
    cleaned = clean_query(query)
    entities = extract_medical_entities(cleaned)
    expanded_terms = expand_synonyms(cleaned, entities)
    vector_query = prepare_vector_query(cleaned)
    keyword_query = prepare_keyword_query(cleaned, expanded_terms)
    filters = extract_filters(cleaned)

    return {
        "original_query": query,
        "cleaned_query": cleaned,
        "entities": entities,
        "expanded_terms": expanded_terms,
        "vector_query": vector_query,
        "keyword_query": keyword_query,
        "filters": filters,
    }



# ==================== 测试代码 ====================

if __name__ == "__main__":
    test_cases = [
        "二甲双胍对心血管疾病有何影响？",
        "  What is  the effect of   MI  ",
        "aspirin与warfarin同时使用会怎样?",
        "",
        "   ",
        "COVID-19的最新治疗方案",
    ]

    print("=== clean_query 测试 ===")
    for tc in test_cases:
        result = clean_query(tc)
        print(f"输入: {tc!r}")
        print(f"输出: {result!r}")
        print("-" * 50)

    print("\n=== extract_medical_entities 测试 ===")
    entity_test_cases = [
        "What is the interaction between Aspirin and warfarin?",
        "metformin对糖尿病患者的作用",
        "ASPIRIN and INSULIN combined therapy",
        "This query has no medical drug entities",
        "aspirin aspirin aspirin",  # 测试去重
    ]
    for tc in entity_test_cases:
        cleaned = clean_query(tc)
        result = extract_medical_entities(cleaned)
        print(f"输入: {tc!r}")
        print(f"识别实体: {result}")
        print("-" * 50)

    print("\n=== expand_synonyms 测试 ===")
    synonym_test_cases = [
        "What is the effect of MI on health",       # 命中缩写mi
        "aspirin and MI risk",                        # 药物实体 + 缩写都有
        "metformin对糖尿病的作用",                     # 无同义词命中
    ]
    for tc in synonym_test_cases:
        cleaned = clean_query(tc)
        ents = extract_medical_entities(cleaned)
        expanded = expand_synonyms(cleaned, ents)
        print(f"输入: {tc!r}")
        print(f"实体: {ents}")
        print(f"扩展同义词: {expanded}")
        print("-" * 50)

    print("\n=== prepare_vector_query / prepare_keyword_query 测试 ===")
    query_test_cases = [
        "What is the effect of MI on health",
        "aspirin and warfarin interaction",
    ]
    for tc in query_test_cases:
        cleaned = clean_query(tc)
        ents = extract_medical_entities(cleaned)
        expanded = expand_synonyms(cleaned, ents)
        vec_q = prepare_vector_query(cleaned)
        kw_q = prepare_keyword_query(cleaned, expanded)
        print(f"输入: {tc!r}")
        print(f"向量检索query: {vec_q!r}")
        print(f"关键词检索query: {kw_q!r}")
        print("-" * 50)

    print("\n=== extract_filters 测试 ===")
    filter_test_cases = [
        "metformin近5年的研究进展",
        "aspirin治疗2020年以后的临床试验",
        "MI research before 2015",
        "warfarin的一般作用机制",  # 无时间过滤条件
    ]
    for tc in filter_test_cases:
        cleaned = clean_query(tc)
        filters = extract_filters(cleaned)
        print(f"输入: {tc!r}")
        print(f"过滤条件: {filters}")
        print("-" * 50)

    print("\n=== process_medical_query 端到端测试 ===")
    e2e_test_cases = [
        "二甲双胍对心血管疾病有何影响？",
        "aspirin近5年对MI风险的影响有哪些研究?",
    ]
    for tc in e2e_test_cases:
        result = process_medical_query(tc)
        print(f"输入: {tc!r}")
        for k, v in result.items():
            print(f"  {k}: {v}")
        print("-" * 50)

    print("\n=== 中文药名识别测试 ===")
    cn_test_cases = [
        "二甲双胍对心血管疾病有何影响？",
        "阿司匹林和华法林一起吃会怎样",
    ]
    for tc in cn_test_cases:
        result = process_medical_query(tc)
        print(f"输入: {tc!r}")
        print(f"实体: {result['entities']}")
        print(f"同义词扩展: {result['expanded_terms']}")
        print("-" * 50)

