"""
LughaGen Tier 1 Prompts — Paraphrase-Based Curriculum
"""
import random
from collections import defaultdict
from config import CURRICULUM_WEIGHTS

STYLES  = list(CURRICULUM_WEIGHTS.keys())
WEIGHTS = list(CURRICULUM_WEIGHTS.values())

_PROMPTS = {
    "kikuyu": {
        "paraphrase_simple": lambda t: (
            f'You are a fluent native Gikuyu speaker from Central Kenya.\n'
            f'Rewrite this sentence in two different ways — same meaning, different words:\n'
            f'"{t}"\n'
            f'Use correct diacritics (i, u). Sound natural, not translated.\n'
            f'Reply ONLY with JSON: {{"v1": "...", "v2": "..."}}'
        ),
        "paraphrase_formal": lambda t: (
            f'You are a Gikuyu scholar writing formal written Kikuyu.\n'
            f'Rewrite this sentence twice in formal, educated Kikuyu:\n'
            f'"{t}"\n'
            f'Correct diacritics required. No English borrowings.\n'
            f'Reply ONLY with JSON: {{"v1": "...", "v2": "..."}}'
        ),
        "spoken": lambda t: (
            f'You are a Gikuyu speaker from Nyeri having a casual conversation.\n'
            f'Rewrite this twice as you would say it naturally in speech:\n'
            f'"{t}"\n'
            f'Use everyday spoken Kikuyu. Diacritics required.\n'
            f'Reply ONLY with JSON: {{"v1": "...", "v2": "..."}}'
        ),
        "idiomatic": lambda t: (
            f'You are an elder Gikuyu speaker who knows traditional expressions.\n'
            f'Rewrite this sentence twice using Kikuyu proverbs or idioms:\n'
            f'"{t}"\n'
            f'Use authentic Kikuyu wisdom expressions. Correct diacritics.\n'
            f'Reply ONLY with JSON: {{"v1": "...", "v2": "..."}}'
        ),
        "dialect_variation": lambda t: (
            f'You are a native Gikuyu speaker.\n'
            f'Rewrite this twice — once as spoken in Kiambu, once in Murang\'a:\n'
            f'"{t}"\n'
            f'Correct diacritics required.\n'
            f'Reply ONLY with JSON: {{"v1": "...", "v2": "..."}}'
        ),
    },
    "kamba": {
        "paraphrase_simple": lambda t: (
            f'You are a fluent native Kikamba speaker from Machakos.\n'
            f'Rewrite this sentence in two different ways — same meaning, different words:\n'
            f'"{t}"\n'
            f'Use correct diacritics (i, u). Sound natural, not translated.\n'
            f'Reply ONLY with JSON: {{"v1": "...", "v2": "..."}}'
        ),
        "paraphrase_formal": lambda t: (
            f'You are a Kamba scholar writing formal written Kikamba.\n'
            f'Rewrite this sentence twice in formal, educated Kamba:\n'
            f'"{t}"\n'
            f'Correct diacritics required. Avoid Swahili or English mixing.\n'
            f'Reply ONLY with JSON: {{"v1": "...", "v2": "..."}}'
        ),
        "spoken": lambda t: (
            f'You are a Kamba speaker from Kitui having a casual conversation.\n'
            f'Rewrite this twice as you would say it in everyday Kamba speech:\n'
            f'"{t}"\n'
            f'Natural spoken Kamba. Diacritics required.\n'
            f'Reply ONLY with JSON: {{"v1": "...", "v2": "..."}}'
        ),
        "idiomatic": lambda t: (
            f'You are an elder Kamba speaker who knows traditional sayings.\n'
            f'Rewrite this sentence twice using Kamba proverbs or idioms:\n'
            f'"{t}"\n'
            f'Use authentic ukambani expressions. Correct diacritics.\n'
            f'Reply ONLY with JSON: {{"v1": "...", "v2": "..."}}'
        ),
        "dialect_variation": lambda t: (
            f'You are a native Kamba speaker.\n'
            f'Rewrite this twice — once as spoken in Machakos, once in Kitui:\n'
            f'"{t}"\n'
            f'Correct diacritics required.\n'
            f'Reply ONLY with JSON: {{"v1": "...", "v2": "..."}}'
        ),
    },
    "luo": {
        "paraphrase_simple": lambda t: (
            f'You are a fluent native Dholuo speaker from Kisumu.\n'
            f'Rewrite this sentence in two different ways — same meaning, different words:\n'
            f'"{t}"\n'
            f'Sound like authentic spoken Dholuo. No Swahili/English structure.\n'
            f'Reply ONLY with JSON: {{"v1": "...", "v2": "..."}}'
        ),
        "paraphrase_formal": lambda t: (
            f'You are a Luo scholar writing formal written Dholuo.\n'
            f'Rewrite this sentence twice in formal, educated Dholuo:\n'
            f'"{t}"\n'
            f'Use complete Luo grammar. Avoid code-switching.\n'
            f'Reply ONLY with JSON: {{"v1": "...", "v2": "..."}}'
        ),
        "spoken": lambda t: (
            f'You are a Luo speaker from Siaya having a casual conversation.\n'
            f'Rewrite this twice as you would naturally say it in Dholuo:\n'
            f'"{t}"\n'
            f'Use everyday Nyanza spoken Luo.\n'
            f'Reply ONLY with JSON: {{"v1": "...", "v2": "..."}}'
        ),
        "idiomatic": lambda t: (
            f'You are a Luo elder who knows traditional Dholuo proverbs.\n'
            f'Rewrite this sentence twice using authentic Luo sayings:\n'
            f'"{t}"\n'
            f'Use traditional Nam Lolwe expressions.\n'
            f'Reply ONLY with JSON: {{"v1": "...", "v2": "..."}}'
        ),
        "dialect_variation": lambda t: (
            f'You are a native Dholuo speaker.\n'
            f'Rewrite this twice — once around Kisumu, once in South Nyanza:\n'
            f'"{t}"\n'
            f'Keep authentic Dholuo structure in both.\n'
            f'Reply ONLY with JSON: {{"v1": "...", "v2": "..."}}'
        ),
    },
    "swahili": {
        "paraphrase_simple": lambda t: (
            f'Wewe ni mzungumzaji wa asili wa Kiswahili sanifu cha Kenya.\n'
            f'Andika upya sentensi hii mara mbili — maana sawa, maneno tofauti:\n'
            f'"{t}"\n'
            f'Tumia Kiswahili cha kawaida cha Kenya.\n'
            f'Jibu kwa JSON tu: {{"v1": "...", "v2": "..."}}'
        ),
        "paraphrase_formal": lambda t: (
            f'Wewe ni mwandishi wa Kiswahili cha rasmi.\n'
            f'Andika upya sentensi hii mara mbili kwa Kiswahili cha rasmi:\n'
            f'"{t}"\n'
            f'Tumia sarufi sahihi. Jibu kwa JSON tu: {{"v1": "...", "v2": "..."}}'
        ),
        "spoken": lambda t: (
            f'Wewe ni mzungumzaji wa Kiswahili unayezungumza na rafiki yako.\n'
            f'Andika upya mara mbili kwa lugha ya mazungumzo ya kila siku:\n'
            f'"{t}"\n'
            f'Jibu kwa JSON tu: {{"v1": "...", "v2": "..."}}'
        ),
        "idiomatic": lambda t: (
            f'Wewe ni mzee anayejua methali na misemo ya Kiswahili.\n'
            f'Andika upya mara mbili ukitumia methali au misemo ya Kiswahili:\n'
            f'"{t}"\n'
            f'Jibu kwa JSON tu: {{"v1": "...", "v2": "..."}}'
        ),
        "dialect_variation": lambda t: (
            f'Wewe ni mzungumzaji wa Kiswahili.\n'
            f'Andika upya mara mbili — mara ya kwanza kama Mswahili wa pwani,\n'
            f'mara ya pili kama mtu wa Nairobi:\n'
            f'"{t}"\n'
            f'Jibu kwa JSON tu: {{"v1": "...", "v2": "..."}}'
        ),
    },
}


class Tier1Curriculum:
    def __init__(self, lang):
        if lang not in _PROMPTS:
            raise ValueError(f"Unknown language: {lang}")
        self.lang   = lang
        self.counts = defaultdict(int)
        self.total  = 0

    def sample_style(self):
        return random.choices(STYLES, weights=WEIGHTS, k=1)[0]

    def build(self, text, style=None):
        if style is None:
            style = self.sample_style()
        if style not in _PROMPTS[self.lang]:
            style = "paraphrase_simple"
        prompt = _PROMPTS[self.lang][style](text)
        self.counts[style] += 1
        self.total += 1
        return prompt, style

    def distribution(self):
        return {s: round(self.counts[s] / max(self.total, 1), 3) for s in STYLES}
