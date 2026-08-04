"""
LughaGen Tier 2 Prompts — v1.2 Final

FIX1: All instructions in English (prevents Swahili output bias)
FIX2: JSON instruction on every prompt (reduces parse failures)
Added: code_switch_ratio() enforcement function
Added: Style balancing + code-switch cap in Tier2Curriculum
"""

import random

# Appended to every prompt — clear, unambiguous JSON requirement
_JSON_INSTRUCTION = (
    "\n\nRespond ONLY with a single line of valid JSON. "
    "No explanation, no markdown, no preamble.\n"
    'Required format: {"v1": "first version here", "v2": "second version here"}'
)

# English stopwords used for post-generation code-switch ratio check
ENGLISH_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "have", "has",
    "i", "you", "he", "she", "we", "they", "and", "or", "but",
    "in", "on", "at", "to", "for", "of", "with", "this", "that",
    "it", "not", "do", "did", "will", "can", "very", "just", "so",
    "be", "been", "being", "my", "your", "his", "her", "our", "their",
}


def code_switch_ratio(text: str) -> float:
    """Returns fraction of English words in text. Used to enforce CODE_SWITCH_WORD_MAX."""
    words = [w.lower().strip(".,!?;:\"'()[]") for w in text.split()]
    words = [w for w in words if len(w) > 1]
    if not words:
        return 0.0
    return sum(1 for w in words if w in ENGLISH_STOPWORDS) / len(words)


# ─────────────────────────────────────────────────────────────────────────────
# KAMBA
# ─────────────────────────────────────────────────────────────────────────────
_KAMBA = {
    "dialogue_conversion": lambda t: (
        "You are a native Kamba speaker from Machakos County, Kenya.\n"
        "Convert the sentence below into two different short natural dialogues "
        "between two Kamba speakers in everyday life (home, farm, or market).\n"
        "Both dialogues must be entirely in Kamba. Each must have at least 2 turns.\n"
        f'Sentence: "{t}"'
        + _JSON_INSTRUCTION
    ),
    "context_expansion": lambda t: (
        "You are a native Kamba speaker from Kitui County, Kenya.\n"
        "Expand the sentence below into two different 2-3 sentence passages "
        "of natural spoken Kamba with real Kenyan context (family, farm, market).\n"
        "Both expansions must be entirely in Kamba.\n"
        f'Sentence: "{t}"'
        + _JSON_INSTRUCTION
    ),
    "code_switching": lambda t: (
        "You are a young Kamba speaker from Nairobi who naturally mixes "
        "Kamba with some Swahili or English when talking casually.\n"
        "Rewrite the sentence in two ways as you would actually say it — "
        "mostly Kamba (70-80%%) with 20-30%% Swahili or English. "
        "Do NOT translate the whole sentence.\n"
        f'Sentence: "{t}"'
        + _JSON_INSTRUCTION
    ),
    "noise_robust_rewrite": lambda t: (
        "You are a Kamba speaker typing a quick WhatsApp or SMS message to a friend.\n"
        "Rewrite the sentence in two different informal casual ways in Kamba — "
        "short, natural, conversational. Abbreviations are fine.\n"
        f'Sentence: "{t}"'
        + _JSON_INSTRUCTION
    ),
    "domain_specific": lambda t: (
        f"You are a {random.choice(['Kamba farmer talking about crops and food', 'Kamba community health worker', 'Kamba primary school teacher'])} "
        "speaking to someone in your village entirely in Kamba.\n"
        "Use the topic from the sentence below to create two different practical conversations "
        f"about {random.choice(['farming and food', 'family health', 'children and education'])}.\n"
        f'Sentence: "{t}"'
        + _JSON_INSTRUCTION
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# KIKUYU
# ─────────────────────────────────────────────────────────────────────────────
_KIKUYU = {
    "dialogue_conversion": lambda t: (
        "You are a native Gikuyu speaker from Kirinyaga County, Kenya.\n"
        "Convert the sentence below into two different short natural dialogues "
        "between two Gikuyu speakers in daily life (home, farm, community meeting).\n"
        "Both dialogues must be entirely in Gikuyu. Each must have at least 2 turns.\n"
        f'Sentence: "{t}"'
        + _JSON_INSTRUCTION
    ),
    "context_expansion": lambda t: (
        "You are a native Gikuyu speaker from Nyeri County, Kenya.\n"
        "Expand the sentence below into two different 2-3 sentence passages "
        "of natural spoken Gikuyu with real Kenyan context (household, farm, community).\n"
        "Both expansions must be entirely in Gikuyu.\n"
        f'Sentence: "{t}"'
        + _JSON_INSTRUCTION
    ),
    "code_switching": lambda t: (
        "You are a young Gikuyu speaker from Nairobi who naturally mixes "
        "Gikuyu with some Swahili or English when speaking to friends.\n"
        "Rewrite the sentence in two ways as you would actually say it — "
        "mostly Gikuyu (70-80%%) with 20-30%% Swahili or English. "
        "Do NOT translate the whole sentence.\n"
        f'Sentence: "{t}"'
        + _JSON_INSTRUCTION
    ),
    "noise_robust_rewrite": lambda t: (
        "You are a Gikuyu speaker typing a quick informal phone message.\n"
        "Rewrite the sentence in two different casual Gikuyu ways — "
        "like a WhatsApp message to a family member. Natural and short.\n"
        f'Sentence: "{t}"'
        + _JSON_INSTRUCTION
    ),
    "domain_specific": lambda t: (
        f"You are a {random.choice(['Gikuyu farmer', 'Gikuyu nurse at a rural clinic', 'Gikuyu secondary school teacher'])} "
        "speaking with someone in your community entirely in Gikuyu.\n"
        "Use the topic from the sentence below to create two different practical conversations "
        f"about {random.choice(['farming and food security', 'health and medicine', 'education for children'])}.\n"
        f'Sentence: "{t}"'
        + _JSON_INSTRUCTION
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# LUO (DHOLUO)
# ─────────────────────────────────────────────────────────────────────────────
_LUO = {
    "dialogue_conversion": lambda t: (
        "You are a native Dholuo speaker from Kisumu County, Kenya.\n"
        "Convert the sentence below into two different short natural dialogues "
        "between two Luo speakers in daily life (home, fishing, community gathering).\n"
        "Both dialogues must be entirely in Dholuo. Each must have at least 2 turns.\n"
        f'Sentence: "{t}"'
        + _JSON_INSTRUCTION
    ),
    "context_expansion": lambda t: (
        "You are a native Dholuo speaker from Siaya County, Kenya.\n"
        "Expand the sentence below into two different 2-3 sentence passages "
        "of natural spoken Dholuo with real Luo context (fishing, farming, village life).\n"
        "Both expansions must be entirely in Dholuo.\n"
        f'Sentence: "{t}"'
        + _JSON_INSTRUCTION
    ),
    "code_switching": lambda t: (
        "You are a young Luo speaker from Nairobi who naturally mixes "
        "Dholuo with some Swahili or English when talking to friends.\n"
        "Rewrite the sentence in two ways as you would actually say it — "
        "mostly Dholuo (70-80%%) with 20-30%% Swahili or English. "
        "Do NOT translate the whole sentence.\n"
        f'Sentence: "{t}"'
        + _JSON_INSTRUCTION
    ),
    "noise_robust_rewrite": lambda t: (
        "You are a Luo speaker typing a quick informal message to a friend.\n"
        "Rewrite the sentence in two different casual Dholuo ways — "
        "like a WhatsApp message. Conversational and short is fine.\n"
        f'Sentence: "{t}"'
        + _JSON_INSTRUCTION
    ),
    "domain_specific": lambda t: (
        f"You are a {random.choice(['Luo fisherman and farmer', 'Luo community health worker', 'Luo village school teacher'])} "
        "speaking with someone in your community entirely in Dholuo.\n"
        "Use the topic from the sentence below to create two different practical conversations "
        f"about {random.choice(['fishing and farming', 'health and family wellbeing', 'education and children'])}.\n"
        f'Sentence: "{t}"'
        + _JSON_INSTRUCTION
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# SWAHILI
# ─────────────────────────────────────────────────────────────────────────────
_SWAHILI = {
    "dialogue_conversion": lambda t: (
        "You are a native Swahili speaker from Nairobi, Kenya.\n"
        "Convert the sentence below into two different short natural dialogues "
        "between two Kenyan Swahili speakers in everyday life.\n"
        "Both dialogues must be in natural Kenyan Swahili. Each must have at least 2 turns.\n"
        f'Sentence: "{t}"'
        + _JSON_INSTRUCTION
    ),
    "context_expansion": lambda t: (
        "You are a Kenyan writer who speaks natural Nairobi Swahili.\n"
        "Expand the sentence below into two different 2-3 sentence passages "
        "of natural Kenyan Swahili with real context (neighbourhood, market, clinic, school).\n"
        f'Sentence: "{t}"'
        + _JSON_INSTRUCTION
    ),
    "code_switching": lambda t: (
        "You are a Nairobi resident who naturally mixes Swahili and English "
        "when speaking (Kenyan code-switching style).\n"
        "Rewrite the sentence in two ways as Nairobians actually speak — "
        "mostly Swahili (70-80%%) with 20-30%% English naturally mixed in. "
        "Do NOT translate the whole sentence.\n"
        f'Sentence: "{t}"'
        + _JSON_INSTRUCTION
    ),
    "noise_robust_rewrite": lambda t: (
        "You are a Kenyan typing a quick informal WhatsApp or Twitter message in Swahili.\n"
        "Rewrite the sentence in two different casual ways — "
        "informal Kenyan Swahili street style. Sheng words and abbreviations are fine.\n"
        f'Sentence: "{t}"'
        + _JSON_INSTRUCTION
    ),
    "domain_specific": lambda t: (
        f"You are a {random.choice(['Kenyan community health officer', 'Kenyan agricultural extension officer', 'Kenyan primary school teacher', 'Kenyan mobile money agent'])} "
        "speaking to a community member in natural Kenyan Swahili.\n"
        "Use the topic from the sentence below to create two different practical conversations "
        f"about {random.choice(['health and disease prevention', 'farming and food security', 'education and school fees', 'mobile money and savings'])} "
        "in a Kenyan context.\n"
        f'Sentence: "{t}"'
        + _JSON_INSTRUCTION
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# REGISTRY
# ─────────────────────────────────────────────────────────────────────────────
_PROMPTS = {
    "kamba":   _KAMBA,
    "kikuyu":  _KIKUYU,
    "luo":     _LUO,
    "swahili": _SWAHILI,
}


class Tier2Curriculum:
    """
    Discourse-driven curriculum with style balancing and code-switch cap.
    Enforces minimum style fractions and caps code-switching frequency.
    """

    def __init__(self, lang: str):
        if lang not in _PROMPTS:
            raise ValueError(f"Unknown language: {lang}")
        self.lang    = lang
        self.counts  = {s: 0 for s in _PROMPTS[lang]}
        self.total   = 0
        self._window = []

    def _under_represented(self):
        from config import STYLE_BALANCE_WINDOW, STYLE_MIN_FRACTION
        if len(self._window) < STYLE_BALANCE_WINDOW:
            return None
        recent = self._window[-STYLE_BALANCE_WINDOW:]
        worst_style, worst_frac = None, 1.0
        for style in self.counts:
            frac = recent.count(style) / STYLE_BALANCE_WINDOW
            if frac < STYLE_MIN_FRACTION and frac < worst_frac:
                worst_style, worst_frac = style, frac
        return worst_style

    def _code_switch_over_cap(self) -> bool:
        from config import STYLE_BALANCE_WINDOW, CODE_SWITCH_MAX_FRACTION
        recent = self._window[-STYLE_BALANCE_WINDOW:] if self._window else []
        if len(recent) < 50:
            return False
        return recent.count("code_switching") / len(recent) > CODE_SWITCH_MAX_FRACTION

    def sample_style(self) -> str:
        from config import STYLES, WEIGHTS
        forced = self._under_represented()
        if forced:
            return forced
        if self._code_switch_over_cap():
            available = [s for s in STYLES if s != "code_switching"]
            w = [WEIGHTS[STYLES.index(s)] for s in available]
            return random.choices(available, weights=w, k=1)[0]
        return random.choices(STYLES, weights=WEIGHTS, k=1)[0]

    def build(self, text: str, style: str = None) -> tuple:
        if style is None:
            style = self.sample_style()
        if style not in _PROMPTS[self.lang]:
            style = "context_expansion"
        prompt = _PROMPTS[self.lang][style](text)
        self.counts[style] += 1
        self._window.append(style)
        self.total += 1
        return prompt, style

    def distribution(self) -> dict:
        return {s: round(c / max(self.total, 1), 3) for s, c in self.counts.items()}

