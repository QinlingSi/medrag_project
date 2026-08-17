"""
scripts/10_llm_generator.py

本地LLM生成模块，封装Ollama调用。
- 支持 system prompt（针对deepseek-r1做特殊处理，避免破坏其<think>模式）
- 支持JSON格式输出要求 + 自动修复常见JSON格式问题
- 支持批量生成
- 兼容新版Ollama：推理过程可能在独立的 'thinking' 字段，也可能嵌在<think>标签里
"""

import json
import re
import time
import requests
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class GenerationResult:
    """单次生成结果"""
    success: bool
    text: str = ""                       # 最终正文（已剥离思考过程）
    json_data: Optional[dict] = None     # 若要求json格式，解析后的结构化结果
    raw_text: str = ""                   # response字段原始内容
    think_content: str = ""              # 提取出的思考过程（如果有）
    error: Optional[str] = None
    elapsed_seconds: float = 0.0
    prompt_tokens_est: int = 0
    completion_tokens_est: int = 0


class LLMGenerator:
    """
    Ollama本地LLM调用封装

    Args:
        model_name: Ollama模型名称，如 "deepseek-r1:7b"
        base_url: Ollama服务地址，默认 http://localhost:11434
        timeout: 请求超时时间（秒）
    """

    NO_SYSTEM_ROLE_MODELS = ("deepseek-r1",)

    def __init__(
        self,
        model_name: str = "deepseek-r1:7b",
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        self.default_temperature = 0.2
        self.default_max_tokens = 1024

        self._test_connection()

    def _test_connection(self):
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            available = [m["name"] for m in resp.json().get("models", [])]
        except Exception as e:
            raise ConnectionError(
                f"无法连接Ollama服务 {self.base_url}，请确认 `ollama serve` 已启动。原始错误: {e}"
            )

        matched = any(self.model_name in m or m in self.model_name for m in available)
        if not matched:
            print(
                f"[警告] 模型 '{self.model_name}' 未在Ollama已拉取列表中找到。\n"
                f"已有模型: {available}\n"
                f"如果确认名称无误可忽略此警告（可能是tag格式差异）。"
            )
        else:
            print(f"[OK] 已连接Ollama，模型 '{self.model_name}' 可用。")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        require_json: bool = False,
    ) -> GenerationResult:
        temperature = self.default_temperature if temperature is None else temperature
        max_tokens = self.default_max_tokens if max_tokens is None else max_tokens

        full_prompt = self._build_full_prompt(prompt, system_prompt, require_json)

        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        start = time.time()
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate", json=payload, timeout=self.timeout
            )
            resp.raise_for_status()
            resp_json = resp.json()
            raw_text = resp_json.get("response", "")
            thinking_field = resp_json.get("thinking", "")
        except Exception as e:
            return GenerationResult(
                success=False,
                error=str(e),
                elapsed_seconds=time.time() - start,
            )
        elapsed = time.time() - start

        # 新版Ollama把推理过程放在独立的thinking字段；老版本嵌在<think>标签里，两种都兼容
        if thinking_field:
            think_content = thinking_field
            clean_text = raw_text
        else:
            think_content, clean_text = self._split_think(raw_text)

        result = GenerationResult(
            success=True,
            text=clean_text.strip(),
            raw_text=raw_text,
            think_content=think_content,
            elapsed_seconds=elapsed,
            prompt_tokens_est=len(full_prompt) // 4,
            completion_tokens_est=(len(raw_text) + len(think_content)) // 4,
        )

        # 正文为空但思考过程不为空，通常是num_predict不够用，思考没结束就被截断
        if not result.text.strip() and think_content:
            result.success = False
            result.error = (
                "response为空：推理过程占满了max_tokens导致被截断（done_reason=length）。"
                "请调大max_tokens再试。"
            )

        if require_json and result.success:
            json_data, err = self._extract_and_fix_json(clean_text)
            if json_data is None:
                result.success = False
                result.error = f"JSON解析失败: {err}"
            else:
                result.json_data = json_data

        return result

    def batch_generate(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        require_json: bool = False,
        verbose: bool = True,
    ) -> List[GenerationResult]:
        results = []
        for i, p in enumerate(prompts):
            if verbose:
                print(f"[batch_generate] {i + 1}/{len(prompts)} ...", end=" ")
            r = self.generate(
                p,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                require_json=require_json,
            )
            if verbose:
                status = "OK" if r.success else f"FAIL({r.error})"
                print(f"{status} ({r.elapsed_seconds:.1f}s)")
            results.append(r)
        return results

    def _build_full_prompt(
        self, prompt: str, system_prompt: Optional[str], require_json: bool
    ) -> str:
        parts = []

        if system_prompt:
            parts.append(f"[系统指令]\n{system_prompt}\n")

        parts.append(prompt)

        if require_json:
            parts.append(
                "\n\n请严格按照JSON格式输出，不要添加任何JSON之外的说明文字、"
                "不要使用markdown代码块包裹（不要```json）。"
            )

        return "\n".join(parts)

    @staticmethod
    def _split_think(raw_text: str):
        match = re.search(r"<think>(.*?)</think>", raw_text, re.DOTALL)
        if match:
            think_content = match.group(1).strip()
            clean_text = raw_text[match.end():].strip()
        else:
            think_content = ""
            clean_text = raw_text
        return think_content, clean_text

    @staticmethod
    def _extract_and_fix_json(text: str):
        candidate = text.strip()

        fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", candidate, re.DOTALL)
        if fence_match:
            candidate = fence_match.group(1).strip()

        first_brace = candidate.find("{")
        last_brace = candidate.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            candidate = candidate[first_brace: last_brace + 1]

        try:
            return json.loads(candidate), None
        except json.JSONDecodeError:
            pass

        fixed = re.sub(r",\s*([\]}])", r"\1", candidate)
        try:
            return json.loads(fixed), None
        except json.JSONDecodeError:
            pass

        open_braces = fixed.count("{") - fixed.count("}")
        open_brackets = fixed.count("[") - fixed.count("]")
        patched = fixed
        if open_brackets > 0:
            patched += "]" * open_brackets
        if open_braces > 0:
            patched += "}" * open_braces
        try:
            return json.loads(patched), None
        except json.JSONDecodeError as e:
            return None, f"{e} | 原始候选文本前200字符: {candidate[:200]}"


if __name__ == "__main__":
    llm = LLMGenerator(model_name="deepseek-r1:7b")
    r = llm.generate("用一句话解释什么是二甲双胍。", temperature=0.2, max_tokens=800)
    print("success:", r.success)
    print("text:", r.text)
    print("think_content长度:", len(r.think_content))
    print("elapsed:", round(r.elapsed_seconds, 1), "s")

