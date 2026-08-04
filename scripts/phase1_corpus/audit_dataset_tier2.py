import json, unicodedata, re, collections, statistics, sys
from pathlib import Path
from datetime import datetime

BASE  = "/workspace/synthetic/tier2"
LANGS = ["kamba", "kikuyu", "luo", "swahili"]

INVALID_DIACRITICS = set('āēīōūķļņģțșłńśźżřěůąęðþæøåßœÿ')
VALID_VOWELS       = set('aeiouãõáéíóúàèìòùâêîôûäëïöüũĩôûîêâ')
NON_LATIN_SCRIPTS  = ['CYRILLIC','ARABIC','CJK','HEBREW','THAI','DEVANAGARI',
                      'HANGUL','GEORGIAN','ARMENIAN','HIRAGANA','KATAKANA',
                      'GREEK','GUJARATI','TAMIL','TELUGU','KANNADA','MALAYALAM']
ENGLISH_STOPWORDS = {'the','and','is','are','was','were','to','of','in','that',
                     'it','for','on','with','as','by','at','from','this','be',
                     'or','an','but','not','have','has','had','do','does','did',
                     'will','would'}
SWAHILI_CORE = {'na','ni','wa','ya','ka','kwa','la','za','pa','mwa','katika',
                'kuwa','pia','au','lakini','kama','hii','hizo','ikiwa','hivyo',
                'kwamba','yote','wote','hata','bali','ili','sana','zaidi'}
LANG_MARKERS = {'kamba':set('ũĩ'),'kikuyu':set('ũĩ'),'luo':set(),'swahili':set()}

def audit_sample(text, lang):
    issues = []
    for c in text:
        if c.isalpha():
            if any(s in unicodedata.name(c,'') for s in NON_LATIN_SCRIPTS):
                issues.append('bad_script'); break
    alpha = [c for c in text if c.isalpha()]
    if alpha:
        latin = sum(1 for c in alpha if 'LATIN' in unicodedata.name(c,''))
        if latin/len(alpha) < 0.60: issues.append('low_latin')
    inv = sum(1 for c in text if c in INVALID_DIACRITICS)
    if inv/max(len(text),1) > 0.03: issues.append('invalid_diacritics')
    if re.search(r'(.)\1{5,}', text): issues.append('char_repeat')
    words = text.split()
    if len(words) > 5:
        single = sum(1 for w in words if len(w)==1 and w.isalpha())
        if single/len(words) > 0.40: issues.append('single_chars')
    if alpha and len(alpha) > 10:
        vowels = sum(1 for c in alpha if c.lower() in VALID_VOWELS)
        if vowels/len(alpha) < 0.12: issues.append('low_vowel')
    words_lower = [w.lower().strip('.,!?;:') for w in words if len(w)>1]
    if words_lower:
        en_hits = sum(1 for w in words_lower if w in ENGLISH_STOPWORDS)
        if en_hits/len(words_lower) > 0.38: issues.append('english_leakage')
    if len(text.strip()) < 15: issues.append('too_short')
    if len(text.strip()) > 800: issues.append('too_long')
    if re.search(r'https?://|www\.', text): issues.append('url')
    if re.search(r'&#\d+;|&[a-z]+;', text): issues.append('html_entities')
    if lang=='swahili' and len(words_lower)>=8:
        sw_hits = sum(1 for w in words_lower if w in SWAHILI_CORE)
        if sw_hits/len(words_lower) < 0.04: issues.append('no_swahili_markers')
    return {
        'issues': issues,
        'length': len(text),
        'word_count': len(words),
        'has_markers': bool(LANG_MARKERS.get(lang) and any(c in LANG_MARKERS[lang] for c in text)),
        'inv_diacritics': inv,
    }

def audit_language(lang, base, sample_size=0):
    path = base / f"{lang}_tier2.jsonl"
    if not path.exists(): return {'error': f"{path} not found"}
    stats = {'total':0,'issues':collections.Counter(),'clean':0,
             'lengths':[],'word_counts':[],'quality_scores':[],
             'styles':collections.Counter(),'has_markers':0,
             'issue_examples':collections.defaultdict(list)}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if sample_size and stats['total'] >= sample_size: break
            stats['total'] += 1
            try:
                d = json.loads(line)
                text = d.get('text','').strip()
                if not text: stats['issues']['empty']+=1; continue
                result = audit_sample(text, lang)
                stats['lengths'].append(result['length'])
                stats['word_counts'].append(result['word_count'])
                stats['quality_scores'].append(d.get('quality',0))
                stats['styles'][d.get('prompt_style','unknown')] += 1
                if result['has_markers']: stats['has_markers'] += 1
                if result['issues']:
                    for issue in result['issues']:
                        stats['issues'][issue] += 1
                        if len(stats['issue_examples'][issue]) < 2:
                            stats['issue_examples'][issue].append(text[:100])
                else:
                    stats['clean'] += 1
            except: stats['issues']['parse_error'] += 1
    return stats

def print_audit(lang, stats):
    if 'error' in stats:
        print(f"\n WARNING: {lang}: {stats['error']}"); return
    total = stats['total']; clean = stats['clean']
    lengths = stats['lengths']; wc = stats['word_counts']; qs = stats['quality_scores']
    clean_pct = clean/max(total,1)*100
    health = 'EXCELLENT' if clean_pct>=99 else 'GOOD' if clean_pct>=97 else 'FAIR' if clean_pct>=95 else 'POOR'
    print(f"\n{'='*68}")
    print(f"  {lang.upper()} AUDIT — {total:,} samples")
    print(f"{'-'*68}")
    print(f"  Overall health   : {health} ({clean_pct:.2f}% clean)")
    print(f"  Clean samples    : {clean:,} / {total:,}")
    print(f"  Flagged samples  : {total-clean:,} ({(total-clean)/max(total,1)*100:.2f}%)")
    if lengths:
        print(f"\n  -- Text Length (chars) --")
        print(f"  Min/Max/Mean/Median : {min(lengths)} / {max(lengths)} / {statistics.mean(lengths):.0f} / {statistics.median(lengths):.0f}")
    if wc:
        print(f"\n  -- Word Count --")
        print(f"  Min/Max/Mean/Median : {min(wc)} / {max(wc)} / {statistics.mean(wc):.1f} / {statistics.median(wc):.0f}")
    if qs:
        print(f"\n  -- Quality Scores --")
        print(f"  Min/Max/Mean/Median : {min(qs):.3f} / {max(qs):.3f} / {statistics.mean(qs):.3f} / {statistics.median(qs):.3f}")
        buckets = collections.Counter()
        for q in qs:
            if   q>=0.95: buckets['0.95-1.00']+=1
            elif q>=0.90: buckets['0.90-0.95']+=1
            elif q>=0.85: buckets['0.85-0.90']+=1
            elif q>=0.80: buckets['0.80-0.85']+=1
            elif q>=0.75: buckets['0.75-0.80']+=1
            else:         buckets['< 0.75']+=1
        for b in ['0.95-1.00','0.90-0.95','0.85-0.90','0.80-0.85','0.75-0.80','< 0.75']:
            n=buckets[b]; bar='#'*min(30,int(n/max(total,1)*300))
            print(f"    {b}  {bar} {n:,} ({n/max(total,1)*100:.1f}%)")
    markers = stats['has_markers']
    if LANG_MARKERS.get(lang):
        print(f"\n  -- Language Markers (u~/i~) --")
        print(f"  Samples with markers : {markers:,} ({markers/max(total,1)*100:.1f}%)")
    if stats['styles']:
        print(f"\n  -- Prompt Style Distribution --")
        for style,count in stats['styles'].most_common():
            bar='#'*min(25,int(count/max(total,1)*250))
            print(f"  {style:<25} {bar} {count:,} ({count/max(total,1)*100:.1f}%)")
    if stats['issues']:
        print(f"\n  -- Issues Detected --")
        for issue,count in stats['issues'].most_common():
            pct=count/max(total,1)*100
            flag='!!' if pct>1.0 else '  '
            print(f"  {flag} {issue:<30} {count:>6,} ({pct:.3f}%)")
        print(f"\n  -- Issue Examples --")
        for issue,examples in list(stats['issue_examples'].items())[:5]:
            print(f"  [{issue}]")
            for ex in examples: print(f"    {ex}")
    else:
        print(f"\n  OK: No issues detected")
    if lengths:
        total_chars=sum(lengths); est_tok=total_chars/4
        print(f"\n  -- Token Estimate --")
        print(f"  Total chars / Est tokens / Avg per sample : {total_chars:,} / {est_tok:,.0f} ({est_tok/1e6:.1f}M) / {est_tok/max(total,1):.0f}")

def main():
    full = '--full' in sys.argv
    sample_size = 0 if full else 10000
    print('='*68)
    print('  LUGHAGEN SYNTHETIC DATASET AUDIT')
    print(f'  {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}')
    print(f'  Mode: {"FULL DATASET" if full else f"SAMPLE ({sample_size:,} per language)"}')
    if not full: print('  Tip: use --full to audit entire dataset')
    print('='*68)
    base = Path(BASE)
    all_stats = []
    grand_total=grand_clean=grand_chars=0
    for lang in LANGS:
        print(f"\n  Auditing {lang}...", end='', flush=True)
        stats = audit_language(lang, base, sample_size=sample_size)
        print(f" {stats.get('total',0):,} samples")
        print_audit(lang, stats)
        all_stats.append((lang,stats))
        if 'error' not in stats:
            grand_total+=stats['total']
            grand_clean+=stats['clean']
            grand_chars+=sum(stats['lengths'])
    print(f"\n{'='*68}")
    print(f"  GRAND SUMMARY")
    print(f"{'-'*68}")
    print(f"  Total samples  : {grand_total:,}")
    print(f"  Clean samples  : {grand_clean:,} ({grand_clean/max(grand_total,1)*100:.2f}%)")
    print(f"  Est. tokens    : {grand_chars/4/1e6:.1f}M")
    print()
    for lang,stats in all_stats:
        if 'error' in stats: continue
        total=stats['total']; clean=stats['clean']
        pct=clean/max(total,1)*100
        chars=sum(stats['lengths']); tokens=chars/4/1e6
        qs=stats['quality_scores']; avg_q=statistics.mean(qs) if qs else 0
        flag='OK' if pct>=99 else 'GOOD' if pct>=97 else 'FAIR' if pct>=95 else 'POOR'
        print(f"  [{flag}] {lang:<10} {total:>7,} samples | {pct:.2f}% clean | avg_q={avg_q:.3f} | ~{tokens:.1f}M tokens")
    overall_pct=grand_clean/max(grand_total,1)*100
    print(f"\n  DATASET QUALITY: {'EXCELLENT - ready for Phase 1' if overall_pct>=99 else 'GOOD' if overall_pct>=97 else 'FAIR - cleaning recommended' if overall_pct>=95 else 'POOR - cleaning required'}")
    print('='*68)

if __name__=='__main__':
    main()
