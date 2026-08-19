from rouge import Rouge


def text_similarity(generated: str, ground_truth: str) -> dict:
    if not generated.strip() or not ground_truth.strip():
        return {"rouge-1": {"r": 0, "p": 0, "f": 0},
                "rouge-2": {"r": 0, "p": 0, "f": 0},
                "rouge-l": {"r": 0, "p": 0, "f": 0}}
    rouge = Rouge()
    scores = rouge.get_scores(generated, ground_truth)[0]
    return scores

import re

MEDICAL_PATTERNS = {
    "percentage": r"\d+(\.\d+)?%",
    "dosage": r"\d+(\.\d+)?\s*(mg|g|ml|毫克|克|毫升)",
    "time_range": r"\d+\s*(天|周|月|年|小时|days?|weeks?|months?|years?|hours?)",
    "safety": r"(风险|副作用|不良反应|risk|side effect|adverse)",
    "treatment": r"(建议|治疗|方案|recommend|treatment|therapy)",
    "mechanism": r"(机制|原理|作用|mechanism)",
}


def extract_key_info(text: str) -> dict:
    return {k: re.findall(p, text) for k, p in MEDICAL_PATTERNS.items()}


def key_info_recall(generated: str, ground_truth: str) -> float:
    gen_info = extract_key_info(generated)
    gt_info = extract_key_info(ground_truth)
    gt_matches = sum(len(v) for v in gt_info.values())
    if gt_matches == 0:
        return 1.0
    overlap = sum(min(len(gen_info[k]), len(gt_info[k])) for k in MEDICAL_PATTERNS)
    return overlap / gt_matches

HALLUCINATION_SIGNALS = [
    r"研究表明",
    r"已被证明",
    r"100%",
    r"完全(安全|有效|无害)",
]

def readability(text: str) -> dict:
    sentences = re.split(r"[。！？.!?]", text)
    sentences = [s for s in sentences if s.strip()]
    lengths = [len(s) for s in sentences]
    return {
        "avg_sentence_length": sum(lengths) / len(lengths) if lengths else 0,
        "num_sentences": len(sentences),
    }

def hallucination_risk(text: str) -> float:
    hits = sum(len(re.findall(p, text)) for p in HALLUCINATION_SIGNALS)
    return min(hits / 3, 1.0)

class AnswerEvaluator:
    def evaluate(self, generated: str, ground_truth: str) -> dict:
        return {
            "similarity": text_similarity(generated, ground_truth),
            "key_info_recall": key_info_recall(generated, ground_truth),
            "hallucination_risk": hallucination_risk(generated),
            "readability": readability(generated),
        }
        
if __name__ == "__main__":
    demo_generated = "Metformin is commonly used to treat type 2 diabetes, with a typical dose of 500mg, risk of mild side effects."
    demo_ground_truth = "Metformin, dosage 500mg, is a first-line treatment for type 2 diabetes with some risk of side effects."

    evaluator = AnswerEvaluator()
    result = evaluator.evaluate(demo_generated, demo_ground_truth)

    import json
    print(json.dumps(result, ensure_ascii=False, indent=2))