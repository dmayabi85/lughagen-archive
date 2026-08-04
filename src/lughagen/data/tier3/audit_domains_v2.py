"""
audit_domains.py — LughaGen Tier 2 Domain Classification Audit

The generator's domain_specific prompt style does NOT persist the domain
label per record. This script infers the domain from text content using a
multilingual keyword dictionary, then ranks domains per language.

Approach:
  1. Curated keyword sets per domain across Swahili, Kikuyu, Kamba, Luo, English
  2. Score each domain_specific record: count unique keyword matches per domain
  3. Assign domain with highest score (≥2 hits); else "uncategorized"
  4. Aggregate per language: count, %, quality, English ratio
  5. Identify top N domains per language + cross-language matrix

Outputs:
  /workspace/synthetic/tier2/domain_audit_v2_report.json
  /workspace/synthetic/tier2/domain_audit_v2_overall.txt
  /workspace/synthetic/tier2/domain_audit_<lang>.txt
"""

import json
import re
import sys
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

TIER2_DIR = Path("/workspace/synthetic/tier2")
REPORT_PATH = TIER2_DIR / "domain_audit_v2_report.json"
LANGUAGES = ["swahili", "kikuyu", "kamba", "luo"]

MIN_KEYWORD_HITS = 2     # Need ≥2 distinct keywords to assign a domain
TOP_N_DISPLAY = 5

# ── DOMAIN KEYWORD DICTIONARY ────────────────────────────────────
# Each domain has multilingual keyword sets. Matching is case-insensitive and
# diacritic-normalized (ĩ→i, ũ→u) so Kikuyu/Kamba diacritics don't escape.

DOMAIN_KEYWORDS = {
    "health": {
        "swahili": {"matibabu", "dawa", "hospitali", "ugonjwa", "afya", "daktari",
                    "uchunguzi", "saratani", "malaria", "VVU", "ukimwi", "mgonjwa",
                    "kliniki", "tiba", "chanjo", "homa", "magonjwa", "wagonjwa",
                    "kifaduro", "kichaa", "uzazi", "ujauzito", "kichwa", "tumbo",
                    "miguu", "macho", "masikio", "kupumua", "moyo", "damu"},
        "kikuyu":  {"murimu", "ndawa", "muruaru", "thibitari", "ndakitari",
                    "warutia", "ngima", "mahuri", "mwiri", "ngoro", "mutwe",
                    "njega", "muruaru", "mathagana", "uthondeki"},
        "kamba":   {"muumu", "ndawa", "thibitari", "ndakitari", "uwau",
                    "kĩwau", "ũũaa", "kĩwau", "mwii", "ngoo", "mutwe"},
        "luo":     {"tuo", "yath", "ohinga", "laktar", "lekruok", "mbasa",
                    "nyamuoyo", "ng'ato matuo", "wich", "ich", "tielo",
                    "wang'", "huma", "ringre", "rem"},
        "english": {"health", "doctor", "hospital", "disease", "medicine", "treatment",
                    "cancer", "diabetes", "HIV", "AIDS", "covid", "vaccine", "vaccination",
                    "clinic", "patient", "symptoms", "diagnosis", "pregnancy", "maternal",
                    "fruits", "vegetables", "healthy", "medical", "nutrition", "diet"},
    },
    "education": {
        "swahili": {"shule", "mwalimu", "mwanafunzi", "elimu", "kufundisha", "chuo",
                    "darasa", "kitabu", "kusoma", "mtihani", "wanafunzi", "walimu",
                    "kusomesha", "shahada", "stashahada", "mafunzo", "kufaulu",
                    "kidato", "msingi", "sekondari", "watoto wa shule", "kwenda shule"},
        "kikuyu":  {"thukuru", "mwarimu", "mwarimũ", "murutwo", "muruti",
                    "thomo", "kirira", "ibuku", "muthomi", "njama"},
        "kamba":   {"sukulu", "mumanyisya", "mumanyiisya", "kisomo", "muisoma",
                    "kuviti", "mumanyi", "kwenda sukulu", "watoto kwenda"},
        "luo":     {"skul", "japuonj", "somo", "puonj", "buk", "japuonjre",
                    "nyathi skul", "kakuro", "puonjre", "kuwo"},
        "english": {"school", "teacher", "student", "education", "class", "classroom",
                    "university", "college", "exam", "examination", "study", "learning",
                    "pupil", "grade", "diploma", "degree", "scholarship", "lesson",
                    "Education Conversation", "education conversation"},
    },
    "agriculture": {
        "swahili": {"kilimo", "shamba", "mkulima", "mahindi", "mboga", "ng'ombe",
                    "mbuzi", "kuku", "mvua", "msimu", "mbegu", "wakulima", "mashamba",
                    "mifugo", "ufugaji", "kondoo", "kahawa", "chai", "viazi", "maharagwe",
                    "ngano", "mtama", "kupanda", "kuvuna", "mavuno"},
        "kikuyu":  {"urimi", "ũrĩmi", "mugunda", "mũgũnda", "murimi", "mũrĩmi",
                    "mbembe", "ng'ombe", "mburi", "mbũri", "kahawa", "majani",
                    "kũrĩma", "kurima", "mũicii"},
        "kamba":   {"uimi", "ĩũĩmwĩ", "muunda", "muũnda", "muimi", "mũĩmi",
                    "mboo", "mbũi", "muviti", "nzou", "kulima", "kũĩma"},
        "luo":     {"pur", "puodho", "japur", "kech", "nyuom", "alot", "dhok",
                    "diel", "rombe", "gweno", "cham", "bel", "puro"},
        "english": {"farming", "agriculture", "farmer", "farm", "crop", "crops",
                    "maize", "livestock", "cattle", "harvest", "rain", "season",
                    "drought", "irrigation", "fertilizer", "seed", "plant", "planting",
                    "coffee", "tea", "wheat", "potato", "vegetable"},
    },
    "religion": {
        "swahili": {"Mungu", "Yesu", "Bwana", "Roho", "kanisa", "biblia", "sala",
                    "ibada", "msikiti", "imani", "dhambi", "wokovu", "Allah",
                    "Kristo", "mtume", "nabii", "malaika", "pepo", "jehanamu",
                    "ubatizo", "mafundisho", "Quran", "Mwathani", "Mwenyezi",
                    "msalaba", "mtakatifu", "Jehova", "Yehova"},
        "kikuyu":  {"Ngai", "Jehovah", "Jehova", "Yesu", "kanitha", "Mbibilia",
                    "Bibilia", "mahoya", "wihokeku", "wĩhokeku", "Kristo", "muthingu",
                    "Mwathani", "Aklĩsto", "Aklisto", "Mwene", "Akristiani"},
        "kamba":   {"Ngai", "Yesu", "kanisa", "Mbivilia", "Mbibilia", "mboya",
                    "Kilisto", "muthingu", "Asa", "Nzambi", "Aklĩsto", "Aklisto",
                    "Mwathani"},
        "luo":     {"Nyasaye", "Yesu", "kanyakla", "Muma", "Buk Maler", "lemo",
                    "yie", "richo", "Kristo", "kanisa", "Lubanga", "Were",
                    "Jakristo", "Jakrist"},
        "english": {"god", "jesus", "christ", "bible", "church", "pray", "prayer",
                    "faith", "religion", "muslim", "christian", "islam", "lord",
                    "holy spirit", "salvation", "sin", "gospel", "scripture",
                    "pastor", "priest", "mosque", "quran", "jehovah", "yahweh"},
    },
    "politics_governance": {
        "swahili": {"rais", "serikali", "bunge", "mbunge", "uchaguzi", "chama",
                    "kura", "sera", "katiba", "haki", "demokrasia", "mwanasiasa",
                    "kaunti", "gavana", "waziri", "kabineti", "rais", "naibu",
                    "tume", "kura ya maoni", "raia", "uongozi", "magavana"},
        "kikuyu":  {"muthamaki", "mũthamaki", "thirikari", "kaunti", "kĩama",
                    "ũthamaki", "ũteti"},
        "kamba":   {"musumbi", "mũsumbĩ", "thiliikali", "kaunti", "ũsumbĩ"},
        "luo":     {"jatelo", "sirikal", "bunge", "county", "yiero", "raila",
                    "ruoth", "ruodhi", "loch", "kanyakla"},
        "english": {"president", "government", "parliament", "election", "vote",
                    "party", "policy", "constitution", "democracy", "minister",
                    "MP", "county", "governor", "politician", "politics", "senate",
                    "senator", "campaign", "ballot", "leader"},
    },
    "business_economy": {
        "swahili": {"biashara", "pesa", "soko", "bei", "kazi", "ajira", "mfanyakazi",
                    "mshahara", "kampuni", "ushuru", "uchumi", "benki", "wafanyabiashara",
                    "wateja", "bidhaa", "huduma", "soko la hisa", "mikopo", "akaunti",
                    "uwekezaji", "faida", "kazini", "shilingi"},
        "kikuyu":  {"wonjoria", "thogora", "mbeca", "ngoroko", "wira", "wĩra",
                    "thoko", "kĩra"},
        "kamba":   {"ndogora", "mbesa", "wia", "wĩa", "uthukumi", "kuumba"},
        "luo":     {"ohala", "pesa", "chiro", "tich", "kompiuter", "bank",
                    "siling", "alufu", "olang'o", "tichgi"},
        "english": {"business", "money", "market", "price", "job", "company",
                    "salary", "economy", "bank", "tax", "trade", "shilling",
                    "investment", "loan", "profit", "customer", "service",
                    "industry", "employment", "shillings"},
    },
    "sports": {
        "swahili": {"mchezo", "mpira", "timu", "soka", "bingwa", "ligi", "uwanja",
                    "mchezaji", "kocha", "michezo", "wachezaji", "ushindi", "mashindano",
                    "Olimpiki", "Harambee Stars"},
        "kikuyu":  {"kibandika", "thanju", "mthithio"},
        "kamba":   {"muthemba", "muvuli", "kĩthembya"},
        "luo":     {"tugo", "tuko", "kipanga", "ligi", "stadium", "Gor Mahia",
                    "AFC Leopards"},
        "english": {"football", "soccer", "team", "league", "match", "champion",
                    "player", "coach", "stadium", "basketball", "tennis",
                    "olympic", "sports", "tournament", "athlete", "championship",
                    "world cup", "premier league"},
    },
    "family_social": {
        "swahili": {"familia", "mama", "baba", "mtoto", "mwana", "ndoa", "mke",
                    "mume", "mzazi", "watoto", "kaka", "dada", "wazazi", "harusi",
                    "ndugu", "shangazi", "mjomba", "babu", "bibi", "mtoto wa kike",
                    "mtoto wa kiume", "shemeji", "binti", "mwanaume"},
        "kikuyu":  {"famĩlĩa", "maitu", "maitũ", "baba", "mwana", "mutumia",
                    "mũtumia", "muthuri", "ciana", "marĩa", "eitu", "eĩtu",
                    "mũka", "mwarĩ", "mũrũ"},
        "kamba":   {"inya", "tata", "mwana", "mundu muka", "syana", "mũka",
                    "mwĩĩtu", "mũendwa", "atĩani"},
        "luo":     {"miyo", "wuoyo", "nyathi", "nyithindo", "chi", "wuon",
                    "owadwa", "miyowa", "wuonwa", "wuod", "nyar", "nyargi",
                    "nyathina", "wuoda", "nyaminwa", "Maria"},
        "english": {"family", "mother", "father", "child", "children", "wife",
                    "husband", "marriage", "wedding", "parent", "son", "daughter",
                    "brother", "sister", "uncle", "aunt", "grandmother",
                    "grandfather", "household"},
    },
    "food_cooking": {
        "swahili": {"chakula", "ugali", "mboga", "kupika", "samaki", "nyama",
                    "wali", "chai", "kahawa", "maziwa", "chapati", "mandazi",
                    "githeri", "pilau", "biryani", "samosa", "mahindi", "viazi",
                    "matunda", "embe", "ndizi", "papai", "machungwa", "chumvi",
                    "sukari", "mafuta"},
        "kikuyu":  {"irio", "ngima", "nyeni", "kuruga", "kũruga", "nyama",
                    "githeri", "mukimo", "njahĩ"},
        "kamba":   {"liu", "ngima", "nyeni", "kuuga", "kũũga", "nyama",
                    "kithyo"},
        "luo":     {"chiemo", "kuon", "alot", "ringo", "rech", "dhok mokwong"},
        "english": {"food", "cooking", "kitchen", "meat", "fish", "vegetable",
                    "tea", "coffee", "rice", "milk", "ugali", "chapati",
                    "bread", "fruit", "banana", "mango", "recipe", "meal",
                    "breakfast", "lunch", "dinner"},
    },
    "transport": {
        "swahili": {"matatu", "basi", "gari", "barabara", "safari", "abiria",
                    "dereva", "stesheni", "uchukuzi", "magari", "ajali", "trafiki",
                    "leseni", "boda boda", "tuk tuk", "ndege", "uwanja wa ndege",
                    "reli", "treni"},
        "kikuyu":  {"ngari", "barabara", "njĩra", "rũgendo", "rugendo", "muthemba"},
        "kamba":   {"ngali", "ngalĩ", "ndia", "kuthi", "kuthĩ", "ngalĩ"},
        "luo":     {"motoka", "dho-yo", "yo", "wuoth", "ofis"},
        "english": {"matatu", "bus", "car", "road", "transport", "passenger",
                    "driver", "vehicle", "station", "highway", "traffic",
                    "license", "boda boda", "tuk tuk", "airport", "flight",
                    "train", "railway", "accident"},
    },
    "legal_justice": {
        "swahili": {"sheria", "mahakama", "jaji", "polisi", "wakili", "uhalifu",
                    "haki", "korti", "hakimu", "shauri", "kesi", "mshtakiwa",
                    "mshtaki", "mwendesha mashtaka", "mashtaka", "rufaa", "hukumu",
                    "magereza", "afisa wa polisi"},
        "kikuyu":  {"watho", "ndũĩka", "thigari", "thingitari"},
        "kamba":   {"mwia", "mwĩa", "thigali", "kothi"},
        "luo":     {"chik", "askari", "kot", "ng'ad bura", "jachik"},
        "english": {"law", "court", "judge", "police", "lawyer", "crime",
                    "justice", "officer", "case", "rights", "prosecution",
                    "defendant", "appeal", "verdict", "sentence", "prison",
                    "advocate", "attorney"},
    },
    "technology_digital": {
        "swahili": {"simu", "tarakilishi", "kompyuta", "intaneti", "mtandao",
                    "tovuti", "programu", "ujumbe", "mtandao wa kijamii",
                    "mitandao", "ujasusi", "akaunti ya mtandao"},
        "kikuyu":  {"thĩmũ", "kompiuta"},
        "kamba":   {"simu", "sĩmu", "kompiuta"},
        "luo":     {"simu", "kompiuter", "wifi", "intaneti"},
        "english": {"phone", "computer", "internet", "online", "website", "app",
                    "digital", "software", "technology", "tech", "data", "wifi",
                    "smartphone", "social media", "facebook", "twitter", "whatsapp",
                    "tiktok", "youtube", "google", "instagram"},
    },
    "weather_climate": {
        "swahili": {"hali ya hewa", "mvua", "jua", "joto", "baridi", "mafuriko",
                    "ukame", "tabia ya nchi", "mabadiliko ya tabianchi"},
        "kikuyu":  {"riũa", "mbura", "mahĩu"},
        "kamba":   {"syua", "mbua"},
        "luo":     {"chieng'", "koth", "oro"},
        "english": {"weather", "rain", "sun", "hot", "cold", "flood", "drought",
                    "climate", "climate change", "temperature", "storm",
                    "heat wave", "rainfall"},
    },
}

# ── METRIC HELPERS ───────────────────────────────────────────────

def normalize_diacritics(s: str) -> str:
    return re.sub(r"[ĩ]", "i", re.sub(r"[ũ]", "u", s.lower()))


def english_function_ratio(text: str) -> float:
    EFW = set("""the and or but to of in for on at with from by is are was were
be been being this that these those i you he she it we they me him her us them""".split())
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    if not words:
        return 0.0
    return sum(1 for w in words if w in EFW) / len(words)


# ── DOMAIN CLASSIFIER ────────────────────────────────────────────

def build_keyword_index():
    """Flatten keyword sets into a single (keyword_lowercased) -> domain map."""
    # Build a per-domain set of normalized keywords (sorted longer-first for
    # multi-word matching).
    domain_kw = {}
    for domain, by_lang in DOMAIN_KEYWORDS.items():
        kws = set()
        for lang_set in by_lang.values():
            for kw in lang_set:
                kws.add(normalize_diacritics(kw))
        domain_kw[domain] = kws
    return domain_kw


DOMAIN_INDEX = build_keyword_index()


def classify_domain(text: str) -> tuple:
    """
    Return (domain_label, hit_count, runner_up_label, runner_up_count).
    Domain is 'uncategorized' if max hits < MIN_KEYWORD_HITS.
    """
    text_norm = normalize_diacritics(text)
    # Use word-boundary regex for single-token keywords; substring for multi-token
    scores = {}
    for domain, kws in DOMAIN_INDEX.items():
        hit_count = 0
        for kw in kws:
            if " " in kw:
                # Multi-word: substring match (after diacritic normalization)
                if kw in text_norm:
                    hit_count += 1
            else:
                # Single word: word boundary
                if re.search(rf"\b{re.escape(kw)}\b", text_norm):
                    hit_count += 1
        if hit_count:
            scores[domain] = hit_count

    if not scores:
        return ("uncategorized", 0, None, 0)

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    top_domain, top_hits = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else (None, 0)

    if top_hits < MIN_KEYWORD_HITS:
        return ("uncategorized", top_hits, runner_up[0], runner_up[1])

    return (top_domain, top_hits, runner_up[0], runner_up[1])


# ── ACCUMULATORS ─────────────────────────────────────────────────

class DomainBucket:
    def __init__(self):
        self.n = 0
        self.qualities = []
        self.eng_ratios = []
        self.lengths = []
        self.examples = []
        self.hit_counts = []

    def add(self, rec, text, hit_count):
        self.n += 1
        self.hit_counts.append(hit_count)
        q = rec.get("quality") or rec.get("quality_score")
        if q is not None:
            try:
                self.qualities.append(float(q))
            except (TypeError, ValueError):
                pass
        self.eng_ratios.append(english_function_ratio(text))
        self.lengths.append(len(text))
        if len(self.examples) < 5 and len(text) >= 50:
            self.examples.append({
                "text": text[:220],
                "hit_count": hit_count,
                "quality": round(float(q), 3) if q else None,
                "english_ratio": round(english_function_ratio(text), 4),
            })

    def summary(self) -> dict:
        return {
            "n": self.n,
            "quality_mean": round(statistics.mean(self.qualities), 4) if self.qualities else None,
            "english_ratio_mean": round(statistics.mean(self.eng_ratios), 4) if self.eng_ratios else 0,
            "length_mean": round(statistics.mean(self.lengths), 1) if self.lengths else 0,
            "hit_count_mean": round(statistics.mean(self.hit_counts), 2) if self.hit_counts else 0,
            "examples": self.examples,
        }


# ── PROCESSING ───────────────────────────────────────────────────

def iter_jsonl(path: Path):
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def analyze_language(lang: str) -> dict:
    print(f"\n  Analyzing {lang}...", flush=True)
    path = TIER2_DIR / f"{lang}_final_dataset_v2.jsonl"
    if not path.exists():
        return {"language": lang, "missing": True}

    domain_stats = defaultdict(DomainBucket)
    n_total_dataset = 0
    n_domain_specific = 0
    runner_up_counter = Counter()  # tracks secondary domain when primary is also assigned

    for rec in iter_jsonl(path):
        n_total_dataset += 1
        if rec.get("prompt_style") != "domain_specific":
            continue
        n_domain_specific += 1
        text = rec.get("text") or ""
        domain, hits, runner, runner_hits = classify_domain(text)
        domain_stats[domain].add(rec, text, hits)
        if runner and runner_hits >= MIN_KEYWORD_HITS:
            runner_up_counter[runner] += 1

        if n_domain_specific % 5000 == 0:
            print(f"    ... {n_domain_specific:,} domain_specific records classified", flush=True)

    print(f"    Done. {n_domain_specific:,} domain_specific records of {n_total_dataset:,} total", flush=True)

    domain_summaries = {d: b.summary() for d, b in domain_stats.items()}

    # Build ranked list
    ranked = sorted(domain_stats.items(), key=lambda x: -x[1].n)
    domain_distribution = []
    for d, b in ranked:
        domain_distribution.append({
            "domain": d,
            "n": b.n,
            "pct_of_domain_specific": round(b.n / n_domain_specific * 100, 2) if n_domain_specific else 0,
            "pct_of_full_dataset":    round(b.n / n_total_dataset * 100, 2) if n_total_dataset else 0,
        })

    return {
        "language":              lang,
        "n_total_dataset":       n_total_dataset,
        "n_domain_specific":     n_domain_specific,
        "domain_distribution":   domain_distribution,
        "domain_summaries":      domain_summaries,
        "runner_up_distribution": dict(runner_up_counter.most_common(10)),
    }


# ── REPORTS ──────────────────────────────────────────────────────

def write_overall_report(all_results: dict):
    path = TIER2_DIR / "domain_audit_v2_overall.txt"

    # Find all domains that appeared anywhere
    all_domains = set()
    for r in all_results.values():
        if not r.get("missing"):
            for d in r["domain_distribution"]:
                all_domains.add(d["domain"])

    # Order: by total volume across languages (excluding "uncategorized" which goes last)
    domain_totals = Counter()
    for r in all_results.values():
        if not r.get("missing"):
            for d in r["domain_distribution"]:
                domain_totals[d["domain"]] += d["n"]
    domain_order = [d for d, _ in domain_totals.most_common() if d != "uncategorized"]
    if "uncategorized" in all_domains:
        domain_order.append("uncategorized")

    with open(path, "w", encoding="utf-8") as f:
        f.write("# LughaGen Tier 2 — DOMAIN AUDIT (domain_specific bucket)\n")
        f.write(f"# Generated: {datetime.utcnow().isoformat()}Z\n")
        f.write(f"# Method: keyword-based classification (≥{MIN_KEYWORD_HITS} distinct keyword hits required)\n\n")

        # Per-language top N
        f.write("=" * 100 + "\n")
        f.write(f"  TOP DOMAINS PER LANGUAGE (within domain_specific bucket)\n")
        f.write("=" * 100 + "\n")
        for lang in LANGUAGES:
            r = all_results.get(lang, {})
            if r.get("missing"):
                continue
            f.write(f"\n  {lang.upper()}   (domain_specific bucket: {r['n_domain_specific']:,} records, "
                    f"{r['n_domain_specific']/r['n_total_dataset']*100:.1f}% of full dataset)\n")
            f.write(f"  {'#':<3} {'Domain':<22} {'Count':>10} {'% of dom.spec':>15} {'% of dataset':>13}\n")
            f.write("  " + "-" * 70 + "\n")
            for i, d in enumerate(r["domain_distribution"][:TOP_N_DISPLAY + 2], 1):
                f.write(f"  {i:<3} {d['domain']:<22} {d['n']:>10,} {d['pct_of_domain_specific']:>14.2f}% "
                        f"{d['pct_of_full_dataset']:>12.2f}%\n")

            # Top 2 highlighted
            top2 = r["domain_distribution"][:2]
            if len(top2) >= 2:
                f.write(f"\n  → Top 2: {top2[0]['domain']} ({top2[0]['pct_of_domain_specific']:.1f}%) "
                        f"and {top2[1]['domain']} ({top2[1]['pct_of_domain_specific']:.1f}%)\n")

        # Cross-language matrix
        f.write("\n\n" + "=" * 100 + "\n")
        f.write("  CROSS-LANGUAGE DOMAIN MATRIX (% of each language's domain_specific bucket)\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"  {'Domain':<22}")
        for lang in LANGUAGES:
            if not all_results.get(lang, {}).get("missing"):
                f.write(f" {lang.capitalize():>12}")
        f.write("\n")
        f.write("  " + "-" * 80 + "\n")
        for domain in domain_order:
            f.write(f"  {domain:<22}")
            for lang in LANGUAGES:
                r = all_results.get(lang, {})
                if r.get("missing"):
                    continue
                d = next((x for x in r["domain_distribution"] if x["domain"] == domain), None)
                if d:
                    f.write(f" {d['pct_of_domain_specific']:>11.2f}%")
                else:
                    f.write(f" {'—':>12}")
            f.write("\n")

        # Quality per domain
        f.write("\n" + "=" * 100 + "\n")
        f.write("  MEAN QUALITY PER DOMAIN PER LANGUAGE\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"  {'Domain':<22}")
        for lang in LANGUAGES:
            if not all_results.get(lang, {}).get("missing"):
                f.write(f" {lang.capitalize():>12}")
        f.write("\n")
        f.write("  " + "-" * 80 + "\n")
        for domain in domain_order:
            f.write(f"  {domain:<22}")
            for lang in LANGUAGES:
                r = all_results.get(lang, {})
                if r.get("missing"):
                    continue
                stats = r["domain_summaries"].get(domain)
                if stats and stats.get("quality_mean") is not None:
                    f.write(f" {stats['quality_mean']:>12.4f}")
                else:
                    f.write(f" {'n/a':>12}")
            f.write("\n")

    return path


def write_per_language_report(lang: str, result: dict):
    path = TIER2_DIR / f"domain_audit_v2_{lang}.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Domain Audit — {lang.upper()}\n")
        f.write(f"# Generated: {datetime.utcnow().isoformat()}Z\n")
        f.write(f"# n_total_dataset: {result['n_total_dataset']:,}\n")
        f.write(f"# n_domain_specific: {result['n_domain_specific']:,}\n\n")

        f.write("=" * 80 + "\n")
        f.write("  DOMAIN DISTRIBUTION (ranked)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"  {'#':<3} {'Domain':<22} {'Count':>10} {'% dom.spec':>13} {'% dataset':>12}\n")
        f.write("  " + "-" * 70 + "\n")
        for i, d in enumerate(result["domain_distribution"], 1):
            f.write(f"  {i:<3} {d['domain']:<22} {d['n']:>10,} {d['pct_of_domain_specific']:>12.2f}% "
                    f"{d['pct_of_full_dataset']:>11.2f}%\n")

        # Per-domain detail with examples
        for d in result["domain_distribution"]:
            domain = d["domain"]
            stats = result["domain_summaries"][domain]
            if stats["n"] == 0:
                continue

            f.write("\n" + "=" * 80 + "\n")
            f.write(f"  DOMAIN: {domain}\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"  Records:                  {stats['n']:,}\n")
            f.write(f"  % of domain_specific:     {d['pct_of_domain_specific']:.2f}%\n")
            f.write(f"  % of full dataset:        {d['pct_of_full_dataset']:.2f}%\n")
            f.write(f"  Mean keyword hits:        {stats['hit_count_mean']}\n")
            f.write(f"  Mean length (chars):      {stats['length_mean']}\n")
            f.write(f"  Mean quality_score:       {stats.get('quality_mean', 'n/a')}\n")
            f.write(f"  Mean English ratio:       {stats['english_ratio_mean']}\n")

            if stats.get("examples"):
                f.write(f"\n  Examples:\n")
                for i, ex in enumerate(stats["examples"], 1):
                    f.write(f"    [{i}] hits={ex['hit_count']}  eng={ex['english_ratio']}  q={ex.get('quality')}\n")
                    f.write(f"        {ex['text']}\n\n")

        # Runner-up domains
        if result.get("runner_up_distribution"):
            f.write("\n" + "=" * 80 + "\n")
            f.write("  SECONDARY DOMAINS (top runner-up classifications)\n")
            f.write("=" * 80 + "\n\n")
            f.write("  When records had multiple domain matches, these were the most common runners-up:\n\n")
            for domain, count in result["runner_up_distribution"].items():
                f.write(f"    {domain:<25} {count:>7,}\n")

    return path


# ── MAIN ─────────────────────────────────────────────────────────

def main():
    if not TIER2_DIR.exists():
        print(f"✗ {TIER2_DIR} not found", file=sys.stderr)
        sys.exit(1)

    print("=" * 80)
    print("  TIER 2 DOMAIN AUDIT — domain_specific bucket classification")
    print(f"  {datetime.utcnow().isoformat()}Z")
    print("=" * 80)
    print(f"\n  Domain dictionary: {len(DOMAIN_KEYWORDS)} domains, "
          f"{sum(len(kws) for d in DOMAIN_INDEX.values() for kws in [d]):,} total keywords")

    all_results = {}
    for lang in LANGUAGES:
        all_results[lang] = analyze_language(lang)

    # Write outputs
    print("\n  Writing reports...")
    report = {
        "version":             "domain_audit_v2.0_expanded",
        "generated_at":        datetime.utcnow().isoformat() + "Z",
        "method":              "keyword-based classification (multilingual)",
        "min_keyword_hits":    MIN_KEYWORD_HITS,
        "domains_in_dictionary": list(DOMAIN_KEYWORDS.keys()),
        "per_language":        all_results,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"  ✓ JSON report: {REPORT_PATH}")

    overall = write_overall_report(all_results)
    print(f"  ✓ Overall: {overall}")

    for lang, r in all_results.items():
        if not r.get("missing"):
            per_lang = write_per_language_report(lang, r)
            print(f"  ✓ {lang}: {per_lang}")

    # ── Console summary ──
    print("\n" + "=" * 100)
    print("  TOP DOMAINS PER LANGUAGE (within domain_specific bucket)")
    print("=" * 100)

    for lang in LANGUAGES:
        r = all_results.get(lang, {})
        if r.get("missing"):
            continue
        print(f"\n  {lang.upper()}   (domain_specific: {r['n_domain_specific']:,} records, "
              f"{r['n_domain_specific']/r['n_total_dataset']*100:.1f}% of dataset)")
        print(f"  {'#':<3} {'Domain':<22} {'Count':>10} {'% dom.spec':>13} {'% dataset':>12}")
        print("  " + "-" * 70)
        for i, d in enumerate(r["domain_distribution"][:TOP_N_DISPLAY + 1], 1):
            print(f"  {i:<3} {d['domain']:<22} {d['n']:>10,} {d['pct_of_domain_specific']:>12.2f}% "
                  f"{d['pct_of_full_dataset']:>11.2f}%")
        top2 = r["domain_distribution"][:2]
        if len(top2) >= 2:
            print(f"\n  → TOP 2: {top2[0]['domain']} ({top2[0]['pct_of_domain_specific']:.1f}%) "
                  f"and {top2[1]['domain']} ({top2[1]['pct_of_domain_specific']:.1f}%)")

    # Cross-language top-2 summary table
    print("\n" + "=" * 100)
    print("  TOP 2 DOMAINS — quick comparison")
    print("=" * 100 + "\n")
    print(f"  {'Language':<14} {'#1 Domain':<22} {'#1 %':>8}  {'#2 Domain':<22} {'#2 %':>8}")
    print("  " + "-" * 90)
    for lang in LANGUAGES:
        r = all_results.get(lang, {})
        if r.get("missing"):
            continue
        top2 = r["domain_distribution"][:2]
        if len(top2) < 2:
            continue
        print(f"  {lang.capitalize():<14} {top2[0]['domain']:<22} {top2[0]['pct_of_domain_specific']:>7.2f}%  "
              f"{top2[1]['domain']:<22} {top2[1]['pct_of_domain_specific']:>7.2f}%")

    print("\n" + "=" * 100)
    print(f"\n  Outputs in {TIER2_DIR}:")
    print(f"    domain_audit_v2_report.json")
    print(f"    domain_audit_v2_overall.txt")
    print(f"    domain_audit_<lang>.txt")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
