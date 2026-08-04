import json, unicodedata, re, collections
from pathlib import Path

BASE       = "/workspace/synthetic/tier1"
LANGS      = ["kamba", "kikuyu", "luo", "swahili"]
TARGET     = 100_000
INPUT_SUFFIX  = "_tier1_backup.jsonl"
OUTPUT_SUFFIX = "_tier1_final.jsonl"

NON_LATIN_SCRIPTS = [
    'CYRILLIC','ARABIC','CJK','HEBREW','THAI','DEVANAGARI',
    'HANGUL','GEORGIAN','ARMENIAN','HIRAGANA','KATAKANA',
    'GREEK','GUJARATI','PUNJABI','TAMIL','TELUGU','KANNADA',
    'MALAYALAM','SINHALA','MYANMAR','KHMER','MONGOLIAN',
    'TIBETAN','ETHIOPIC','CHEROKEE','CANADIAN',
]

def has_bad_script(text):
    for c in text:
        if c.isalpha():
            if any(s in unicodedata.name(c, '') for s in NON_LATIN_SCRIPTS):
                return True
    return False

def latin_ratio(text):
    alpha = [c for c in text if c.isalpha()]
    if not alpha: return 0.0
    latin = sum(1 for c in alpha if 'LATIN' in unicodedata.name(c, ''))
    return latin / len(alpha)

INVALID_DIACRITICS = set('āēīōūķļņģțșłńśźżřěůąęðþæøåßœÿ')

def is_garbled(text):
    if not text: return True, "empty"
    invalid_count = sum(1 for c in text if c in INVALID_DIACRITICS)
    if len(text) > 0 and invalid_count / len(text) > 0.03:
        return True, f"invalid_diacritics:{invalid_count}"
    non_alnum = sum(1 for c in text if not c.isalnum() and not c.isspace())
    if len(text) > 0 and non_alnum / len(text) > 0.35:
        return True, "excessive_symbols"
    if re.search(r'(.)\1{5,}', text):
        return True, "char_repetition"
    words = text.split()
    if len(words) > 5:
        single_char = sum(1 for w in words if len(w) == 1 and w.isalpha())
        if single_char / len(words) > 0.40:
            return True, "single_char_words"
    letters = [c for c in text if c.isalpha()]
    if len(letters) > 10:
        vowels = sum(1 for c in letters if c.lower() in 'aeiouãõáéíóúàèìòùâêîôûäëïöüũĩ')
        if vowels / len(letters) < 0.18:
            return True, f"low_vowel_ratio:{vowels/len(letters):.2f}"
    return False, ""

ENGLISH_STOPWORDS = {
    'the','and','is','are','was','were','to','of','in','that','it',
    'for','on','with','as','by','at','from','this','be','or','an',
    'but','not','have','has','had','do','does','did','will','would',
    'could','should','may','might','shall','can','i','you','he','she',
    'we','they','me','him','her','us','them','my','your','his','our',
    'their','what','which','who','where','when','why','how','all',
    'been','being','if','its','than','then','so','up','out','about',
}

def english_leakage(text, threshold=0.38):
    words = [w.lower().strip('.,!?;:()[]{}"\'') for w in text.split()]
    words = [w for w in words if len(w) > 1]
    if len(words) < 5: return False
    hits = sum(1 for w in words if w in ENGLISH_STOPWORDS)
    return (hits / len(words)) > threshold

FOREIGN_MARKERS = {
    'french':     {'le','la','les','un','une','des','est','sont','avec','pour','dans','que','qui','pas','mais','sur','par','au','du','se','je','tu','il','nous','vous','ils','elle','mon','ton','son'},
    'spanish':    {'el','la','los','las','un','una','es','son','con','para','por','que','del','al','se','yo','de','no','si','pero','como','este','esta','mi','su','nos','ellos','me','te','le'},
    'portuguese': {'os','as','um','nao','com','para','por','que','dos','das','ao','da','do','se','em','ele','ela','sao','mas','mais','este','esta','isso','muito','bem','ja','ainda','so'},
    'italian':    {'il','lo','gli','le','uno','sono','con','per','che','del','al','si','io','lui','lei','noi','voi','loro','non','ma','anche','come','questo','questa','ho','ha','mi','ti'},
    'german':     {'der','die','das','ein','eine','ist','sind','mit','fur','von','zu','den','dem','des','bei','nach','auch','noch','wenn','aber','oder','ich','du','er','sie','wir','nicht'},
    'dutch':      {'het','een','met','voor','van','dat','die','op','aan','bij','ook','niet','als','er','ze','we','je','ik','zij','hij','hun','dit','deze','wat','hoe','waar'},
    'swedish':    {'och','att','det','som','for','med','till','pa','av','han','hon','vi','de','ar','var','har','inte','men','om','ett','sig','ska','kan','vill','maste'},
    'norwegian':  {'og','at','det','som','for','med','til','pa','av','han','hun','vi','er','var','har','ikke','men','om','et','seg','skal','kan','vil','ma','bor'},
    'danish':     {'og','at','det','som','for','med','til','pa','af','han','hun','vi','er','var','har','ikke','men','om','et','sig','skal','kan','vil','ma'},
    'polish':     {'jest','sie','nie','to','ze','jak','przez','ale','czy','jego','jej','ich','byc','miec','ktory','ktora','ktore','tego','tej','tym'},
    'czech':      {'jsou','se','to','ze','jak','ale','nebo','pro','pri','jeho','jeji','jejich','byt','mit','ktery','ktera','ktere','jsem','jsi','jsme'},
    'croatian':   {'je','su','se','to','sto','kako','ali','ili','za','na','od','do','sa','pri','zbog','vec','samo'},
    'turkish':    {'bir','bu','ve','da','de','ile','icin','gibi','ama','ya','ne','degil','var','yok','olan','ki','ben','sen','biz','siz','onlar'},
    'indonesian': {'yang','dan','di','ini','itu','dengan','untuk','dari','tidak','ada','saya','kita','kami','mereka','adalah','akan','bisa','juga','sudah'},
    'tagalog':    {'ang','ng','sa','na','at','ay','mga','ko','mo','niya','namin','ninyo','nila','ito','iyon','siya','sila','kami','tayo','kayo','ba','pa'},
    'malay':      {'yang','dan','di','ini','itu','dengan','untuk','dari','tidak','ada','saya','kita','kami','mereka','adalah','akan','boleh','juga','sudah'},
    'zulu':       {'ukuthi','ukuba','noma','kodwa','futhi','ngoba','uma','lapha','khona','manje','njalo','zonke','bonke','yonke','wonke'},
    'xhosa':      {'ukuba','noba','kodwa','futhi','ngoba','ukuthi','ukuze','ngoko','kwaye','kananjalo'},
    'yoruba':     {'ati','fun','nitori','bi','ti','ni','si','pelu','lati','ninu','lori','tabi','sugbon'},
    'igbo':       {'na','ya','ka','ma','mana','ha','anyi','gi','nke','no','ebe','mgbe','ihe','ndi','maka'},
    'hausa':      {'da','don','amma','kuma','ko','sai','idan','wanda','wata','wannan','wadannan','shi','ita','su','mu'},
    'somali':     {'iyo','oo','ah','in','la','ka','ku','uu','ay','waa','aan','leh','kale','sida','markaa','laakiin'},
    'wolof':      {'ak','bi','yi','bu','li','ci','di','mu','nu','ko','leen','ngir','fee','fii','foo','nit','xam'},
    'vietnamese': {'va','cua','la','trong','co','voi','duoc','nhung','cho','nay','do','khong','tu','mot','cac'},
    'hindi_rom':  {'hai','hain','tha','thi','the','mein','ko','ke','ki','ka','se','par','aur','ya','lekin'},
}

TARGET_WHITELIST = {
    'kamba':   {'na','ni','wa','ya','ka','ma','la','ta','ku','mu','si','vi','tu','mi','sa','ki','a','o','e','i','u'},
    'kikuyu':  {'na','ni','wa','ya','ka','ma','la','ta','ku','mu','si','vi','tu','mi','no','ro','gwa','a','o','e','i','u'},
    'luo':     {'gi','mi','ni','bi','di','pi','ti','si','li','ri','ki','ma','ka','la','wa','ya','na','to','go','do','ko','lo','a','o','e','i','u'},
    'swahili': {'na','ni','wa','ya','ka','ma','la','ta','ku','mu','si','vi','tu','mi','za','ji','li','ki','pa','mwa','pia','au','kwa','kama','sana','a','o','e','i','u'},
}

def is_contaminated(text, lang):
    words = [w.lower().strip('.,!?;:()[]{}"\'') for w in text.split()]
    words = [w for w in words if len(w) > 1]
    if len(words) < 5: return False, ""
    whitelist = TARGET_WHITELIST.get(lang, set())
    for foreign_lang, markers in FOREIGN_MARKERS.items():
        hits = sum(1 for w in words if w in markers and w not in whitelist)
        hit_rate = hits / len(words)
        if hit_rate > 0.30 or (hits >= 3 and len(words) >= 8):
            return True, foreign_lang
    return False, ""

def structural_issues(text):
    issues = []
    if len(text.strip()) < 15: issues.append('too_short')
    if len(text.strip()) > 800: issues.append('too_long')
    if re.search(r'https?://|www\.|@\w+\.\w+', text): issues.append('url')
    if re.search(r'\b(def |class |import |return |function\()', text): issues.append('code')
    digits = sum(c.isdigit() for c in text)
    if len(text) > 0 and digits / len(text) > 0.30: issues.append('excessive_digits')
    words = text.split()
    if len(words) >= 8:
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
        repeated = [b for b in bigrams if bigrams.count(b) > 1]
        if len(repeated) > 5: issues.append('repeated_phrases')
    return issues

STRUCTURAL_REJECT = {'too_short', 'url', 'code'}

SWAHILI_CORE = {
    'na','ni','wa','ya','ka','kwa','la','za','pa','mwa','katika','kuwa',
    'pia','au','lakini','kama','hii','hizo','ikiwa','hivyo','kwamba',
    'yote','wote','hata','bali','ili','sana','zaidi','kabla','baada',
    'wakati','jambo','mambo','watu','mtu','siku','mwaka',
}

def fails_language_check(text, lang):
    words = [w.lower().strip('.,!?;:') for w in text.split()]
    if len(words) < 8: return False
    if lang == 'swahili':
        hits = sum(1 for w in words if w in SWAHILI_CORE)
        if hits / len(words) < 0.04: return True
    return False

print("=" * 65)
print("  LUGHAGEN SYNTHETIC DATA CLEANER — v3 Final")
print("=" * 65)

grand_total = grand_kept = 0

for lang in LANGS:
    input_path  = Path(BASE) / f"{lang}{INPUT_SUFFIX}"
    output_path = Path(BASE) / f"{lang}{OUTPUT_SUFFIX}"

    if not input_path.exists():
        input_path = Path(BASE) / f"{lang}_tier1.jsonl"
        if not input_path.exists():
            print(f"WARNING: {lang}: no input file found — skipping")
            continue
        print(f"  {lang}: using current file (no backup found)")

    stats   = collections.Counter()
    samples = {}

    with open(input_path) as fin, open(output_path, 'w') as fout:
        for line in fin:
            line = line.strip()
            if not line: continue
            stats['total'] += 1
            try:
                d    = json.loads(line)
                text = d.get('text', '').strip()
                if not text:
                    stats['empty'] += 1
                    continue
                if has_bad_script(text):
                    stats['bad_script'] += 1
                    samples.setdefault('bad_script', text[:80])
                    continue
                if latin_ratio(text) < 0.60:
                    stats['low_latin'] += 1
                    continue
                garbled, reason = is_garbled(text)
                if garbled:
                    stats['garbled'] += 1
                    stats[f'garbled:{reason}'] += 1
                    samples.setdefault('garbled', f"[{reason}] {text[:80]}")
                    continue
                if english_leakage(text):
                    stats['english'] += 1
                    samples.setdefault('english', text[:80])
                    continue
                contaminated, flang = is_contaminated(text, lang)
                if contaminated:
                    stats['foreign'] += 1
                    stats[f'foreign:{flang}'] += 1
                    samples.setdefault(f'foreign:{flang}', text[:80])
                    continue
                issues = structural_issues(text)
                if any(i in STRUCTURAL_REJECT for i in issues):
                    stats['structural'] += 1
                    continue
                if fails_language_check(text, lang):
                    stats['lang_check'] += 1
                    continue
                fout.write(line + '\n')
                stats['kept'] += 1
            except:
                stats['parse_error'] += 1

    total   = stats['total']
    kept    = stats['kept']
    removed = total - kept
    grand_total += total
    grand_kept  += kept

    print(f'\n{"=" * 65}')
    print(f'  {lang.upper()}: {total:,} -> {kept:,} kept ({removed:,} removed, {removed/max(total,1)*100:.1f}%)')
    print(f'    bad script:     {stats["bad_script"]:>7,}')
    print(f'    low latin:      {stats["low_latin"]:>7,}')
    print(f'    garbled:        {stats["garbled"]:>7,}')
    garbled_types = {k.split(":",1)[1]:v for k,v in stats.items() if k.startswith("garbled:") and ":" in k}
    for r,c in sorted(garbled_types.items(), key=lambda x:-x[1]):
        print(f'      -> {r:<30} {c:>5,}')
    print(f'    english:        {stats["english"]:>7,}')
    print(f'    foreign:        {stats["foreign"]:>7,}')
    foreign_types = {k.split(":",1)[1]:v for k,v in stats.items() if k.startswith("foreign:") and ":" in k}
    for fl,c in sorted(foreign_types.items(), key=lambda x:-x[1])[:8]:
        print(f'      -> {fl:<30} {c:>5,}')
    print(f'    structural:     {stats["structural"]:>7,}')
    print(f'    lang check:     {stats["lang_check"]:>7,}')
    if samples:
        print(f'  Examples:')
        for k,v in list(samples.items())[:3]:
            print(f'    [{k}]: {v}')
    gap = TARGET - kept
    if gap > 0:
        print(f'  WARNING: {gap:,} short of {TARGET:,} target')
    else:
        print(f'  OK: Target met ({kept:,} >= {TARGET:,})')

print(f'\n{"=" * 65}')
print(f'  TOTAL: {grand_total:,} -> {grand_kept:,} kept ({grand_total-grand_kept:,} removed)')
print(f'{"=" * 65}')
print('\nTo replace current files with clean versions run:')
print('  for lang in kamba kikuyu luo swahili; do')
print(f'    mv /workspace/synthetic/tier1/${{lang}}_tier1_final.jsonl /workspace/synthetic/tier1/${{lang}}_tier1.jsonl')
print('  done')
