from dataclasses import dataclass, field
from typing import Optional, TypeVar
from datetime import datetime
import re
import tiktoken
from openai import OpenAI
from loguru import logger
import json
RawPaperItem = TypeVar('RawPaperItem')

@dataclass
class Paper:
    source: str
    title: str
    authors: list[str]
    abstract: str
    url: str
    pdf_url: Optional[str] = None
    full_text: Optional[str] = None
    tldr: Optional[str] = None
    affiliations: Optional[list[str]] = None
    score: Optional[float] = None
    # === 新增字段：中文结构化分析 ===
    chinese_title: Optional[str] = None
    chinese_abstract: Optional[str] = None
    innovation: Optional[str] = None
    datasets: Optional[list[str]] = field(default_factory=list)
    has_code: bool = False
    code_url: Optional[str] = None
    recommendation_reason: Optional[str] = None
    worth_reading: int = 3

    def _generate_chinese_analysis_with_llm(self, openai_client: OpenAI, llm_params: dict):
        """用 LLM 生成结构化的中文分析"""
        lang = llm_params.get('language', 'Chinese')

        prompt = f"""你是一位医学影像/CV领域的资深研究者。请仔细阅读以下论文信息，生成一份结构化的中文分析。

论文标题: {self.title}
论文摘要: {self.abstract}
"""
        if self.full_text:
            full_preview = self.full_text[:4000]
            prompt += f"""论文正文预览（前4000字符）:
{full_preview}
"""

        prompt += f"""
请按以下 JSON 格式输出（严格输出 JSON，不要有其他文字）：

{{
    "chinese_title": "将论文标题翻译成中文",
    "chinese_abstract": "用中文总结论文核心内容（150-300字）",
    "innovation": "论文的创新点（列出1-3条，用中文）",
    "datasets": ["使用的数据集名称1", "数据集2"],
    "has_code": true或false,
    "code_url": "如果有开源代码则填URL，否则填null",
    "recommendation_reason": "为什么这篇论文值得关注（1-2句话，中文）",
    "worth_reading": 1到5的整数，5表示非常值得精读
}}

注意：
- chinese_title: 专业准确的中文翻译
- innovation: 提取真正的创新点，不要泛泛而谈
- datasets: 从论文中提取使用的数据集名称（如CCTA、MIMIC、UK Biobank等），没有则返回空列表
- has_code: 论文是否提供了代码仓库链接（GitHub等）
- worth_reading: 综合论文质量、创新性、与医学影像/CV领域相关性打分

请直接输出JSON，不要有任何markdown代码块标记。"""

        enc = tiktoken.encoding_for_model("gpt-4o")
        prompt_tokens = enc.encode(prompt)
        prompt_tokens = prompt_tokens[:6000]
        prompt = enc.decode(prompt_tokens)

        response = openai_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "你是一位专业的中文学术论文分析助手。你擅长提取论文核心信息、翻译学术标题、评估论文价值。你只输出JSON格式，不输出其他内容。",
                },
                {"role": "user", "content": prompt},
            ],
            **llm_params.get('generation_kwargs', {})
        )
        result = response.choices[0].message.content

        result = re.sub(r'^```(?:json)?\s*', '', result.strip())
        result = re.sub(r'\s*```$', '', result.strip())

        try:
            data = json.loads(result)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', result, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
            else:
                logger.warning(f"Failed to parse LLM response as JSON for {self.url}")
                return

        self.chinese_title = data.get('chinese_title', '')
        self.chinese_abstract = data.get('chinese_abstract', '')
        self.innovation = data.get('innovation', '')
        self.datasets = data.get('datasets', [])
        self.has_code = data.get('has_code', False)
        self.code_url = data.get('code_url')
        self.recommendation_reason = data.get('recommendation_reason', '')
        self.worth_reading = data.get('worth_reading', 3)

        self.tldr = self.chinese_abstract

    def generate_chinese_analysis(self, openai_client: OpenAI, llm_params: dict):
        """生成中文结构化分析，失败时回退"""
        try:
            self._generate_chinese_analysis_with_llm(openai_client, llm_params)
        except Exception as e:
            logger.warning(f"Failed to generate Chinese analysis for {self.url}: {e}")
            self.tldr = self.abstract[:200] if self.abstract else "分析失败"

    def _generate_tldr_with_llm(self, openai_client: OpenAI, llm_params: dict) -> str:
        lang = llm_params.get('language', 'English')
        prompt = f"Given the following information of a paper, generate a one-sentence TLDR summary in {lang}:\n\n"
        if self.title:
            prompt += f"Title:\n {self.title}\n\n"
        if self.abstract:
            prompt += f"Abstract: {self.abstract}\n\n"
        if self.full_text:
            prompt += f"Preview of main content:\n {self.full_text}\n\n"
        if not self.full_text and not self.abstract:
            logger.warning(f"Neither full text nor abstract is provided for {self.url}")
            return "Failed to generate TLDR. Neither full text nor abstract is provided"

        enc = tiktoken.encoding_for_model("gpt-4o")
        prompt_tokens = enc.encode(prompt)
        prompt_tokens = prompt_tokens[:4000]
        prompt = enc.decode(prompt_tokens)

        response = openai_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": f"You are an assistant who perfectly summarizes scientific paper. Your answer should be in {lang}.",
                },
                {"role": "user", "content": prompt},
            ],
            **llm_params.get('generation_kwargs', {})
        )
        tldr = response.choices[0].message.content
        return tldr

    def generate_tldr(self, openai_client: OpenAI, llm_params: dict) -> str:
        try:
            tldr = self._generate_tldr_with_llm(openai_client, llm_params)
            self.tldr = tldr
            return tldr
        except Exception as e:
            logger.warning(f"Failed to generate tldr of {self.url}: {e}")
            tldr = self.abstract
            self.tldr = tldr
            return tldr

    def _generate_affiliations_with_llm(self, openai_client: OpenAI, llm_params: dict) -> Optional[list[str]]:
        if self.full_text is not None:
            prompt = f"Given the beginning of a paper, extract the affiliations of the authors in a python list format, which is sorted by the author order. If there is no affiliation found, return an empty list '[]':\n\n{self.full_text}"
            enc = tiktoken.encoding_for_model("gpt-4o")
            prompt_tokens = enc.encode(prompt)
            prompt_tokens = prompt_tokens[:2000]
            prompt = enc.decode(prompt_tokens)
            affiliations = openai_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an assistant who perfectly extracts affiliations of authors from a paper. You should return a python list of affiliations sorted by the author order. If an affiliation is consisted of multi-level affiliations, you should return the top-level affiliation only. Do not contain duplicated affiliations. If there is no affiliation found, you should return an empty list [ ]. You should only return the final list of affiliations.",
                    },
                    {"role": "user", "content": prompt},
                ],
                **llm_params.get('generation_kwargs', {})
            )
            affiliations = affiliations.choices[0].message.content
            affiliations = re.search(r'\[.*?\]', affiliations, flags=re.DOTALL).group(0)
            affiliations = json.loads(affiliations)
            affiliations = list(set(affiliations))
            affiliations = [str(a) for a in affiliations]
            return affiliations

    def generate_affiliations(self, openai_client: OpenAI, llm_params: dict) -> Optional[list[str]]:
        try:
            affiliations = self._generate_affiliations_with_llm(openai_client, llm_params)
            self.affiliations = affiliations
            return affiliations
        except Exception as e:
            logger.warning(f"Failed to generate affiliations of {self.url}: {e}")
            self.affiliations = None
            return None


@dataclass
class CorpusPaper:
    title: str
    abstract: str
    added_date: datetime
    paths: list[str]
