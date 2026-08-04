"""
Tokenizer fertility measurement for LLaMA-3.1 across Swahili and English.

Produces the token-per-word ratios cited in §1.2, §5.5, §6.1 of the proposal.
The 'fertility' metric (Rust et al., 2021; Ahia et al., 2021) is total tokens
divided by total whitespace-delimited words — lower is better.

Requirements: transformers, huggingface_hub login, LLaMA-3.1 access approved.
Run:  python tokenizer_fertility.py

Saves a small CSV (`fertility_results.csv`) and prints a summary table.
"""

import csv
from transformers import AutoTokenizer


# ─────────────────────────────────────────────────────────────────
# Sample texts — replace with your own representative corpus excerpts
# if available. These are short news-style sentences.
# ─────────────────────────────────────────────────────────────────
SAMPLES = {
    "Swahili": [
        "Habari za asubuhi jinsi gani leo hii",
        "Waziri wa Mambo ya Nje wa China amesema kuwa kuna umuhimu wa kuongeza ushirikiano",
        "Mtoto mdogo anasoma kitabu chake cha hisabati nyumbani kwao kila siku jioni",
        "Tunapenda kusoma vitabu vya hadithi za kale ambavyo vinaeleza utamaduni wetu wa Kiafrika",
        "Wizara ya Afya imetangaza kampeni mpya ya chanjo dhidi ya magonjwa ya watoto",
        "Wanafunzi wa shule ya msingi wanahitaji vifaa vya kujifunzia ili waweze kufanikiwa masomoni",
    ],
    "Kikuyu": [
        "Andũ othe nĩ marĩ na ihooto cia kũmenyera mwĩciro wao",
        "Mwarĩmũ aratheeka athomi ake nĩ ũndũ wa kĩoro kĩao",
        "Twĩthamakaga thĩinĩ wa bũrũri witũ wa Kenya",
    ],
    "Luo": [
        "Jowuoyi gi gimoro amora ma gigombo timo",
        "Wuon-gi nigi yamo maber kuom nyithinde duto",
        "Wadhi e skul mondo wapuonjre weche manyien",
    ],
    "English": [
        "Good morning, how are you today",
        "The Minister of Foreign Affairs of China has said there is a need to strengthen cooperation",
        "A small child is reading her mathematics book at home every evening",
        "We love reading ancient story books that describe our African culture",
        "The Ministry of Health has announced a new vaccination campaign against childhood diseases",
        "Primary school students need learning materials to be able to succeed in their studies",
    ],
}


def fertility(texts, tokenizer):
    """Return (tokens-per-word, total tokens, total words)."""
    total_tokens, total_words = 0, 0
    for t in texts:
        toks = tokenizer.encode(t, add_special_tokens=False)
        words = t.split()
        total_tokens += len(toks)
        total_words  += len(words)
    return total_tokens / total_words, total_tokens, total_words


print("Loading LLaMA-3.1-8B tokenizer ...")
tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B", use_fast=True)

print("\n" + "=" * 70)
print("  LLaMA-3.1 Default Tokenizer Fertility")
print("=" * 70)
print(f"  {'Language':<10} {'Tokens':>8} {'Words':>8} {'Fertility':>11}  Multiplier (vs English)")
print("  " + "-" * 65)

rows = []
en_fert = None
for lang, texts in SAMPLES.items():
    f, tk, wd = fertility(texts, tok)
    if lang == "English":
        en_fert = f
    rows.append((lang, tk, wd, f))

for lang, tk, wd, f in rows:
    mult = (f / en_fert) if en_fert else float("nan")
    flag = "  ← " + ("baseline" if lang == "English" else f"{mult:.2f}× English")
    print(f"  {lang:<10} {tk:>8} {wd:>8} {f:>11.3f}{flag}")

# Write CSV for the proposal appendix / supervisor review
with open("fertility_results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["language", "tokens", "words", "fertility_tokens_per_word"])
    for r in rows:
        w.writerow(r)

print("\n  Results saved to fertility_results.csv")
print("=" * 70)
