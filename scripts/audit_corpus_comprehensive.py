"""
audit_corpus_comprehensive.py

Full audit (no sampling by default) that also EXTRACTS contaminated records for inspection.
Scans every record, categorizes issues, and dumps sample bad records per category.
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
import time


PATTERNS = {
    'high_emoji': re.compile(
        r'[\U0001F300-\U0001F9FF\U0001F600-\U0001F64F\u2600-\u26FF]'
    ),
    'kenyan_phone': re.compile(r'\b(07\d{8}|\+254\d{9})\b'),
    'email': re.compile(r'\b[\w.-]+@[\w.-]+\.[a-z]{2,}\b'),
    'possible_kenyan_id': re.compile(r'(?<![\w\d])\d{8}(?![\w\d])'),
    'ocr_failure': re.compile(r'(llll|rnrn|0O0O|1I1I)'),
    'mojibake': re.compile(r'(\xc3[\xa9\xa8\xa4]|\xe2\x80[\x99\x9c\x9d]|\xef\xbb\xbf)'),
    'within_doc_repetition': re.compile(r'(\b\w+\b)(?:\s+\1){4,}'),
    'long_char_repeat': re.compile(r'(.)\1{15,}'),
}


def classify_record(text):
    if not text or not text.strip():
        return set()
    issues = set()
    text_len = len(text)

    emoji_matches = PATTERNS['high_emoji'].findall(text)
    if len(emoji_matches) > text_len * 0.05:
        issues.add('high_emoji')

    symbol_count = sum(1 for c in text if c in '%%$$^^***@@||~~')
    if text_len > 0 and symbol_count / text_len > 0.1:
        issues.add('high_symbol')

    for issue_name in ['kenyan_phone', 'email', 'possible_kenyan_id',
                       'ocr_failure', 'mojibake', 'within_doc_repetition',
                       'long_char_repeat']:
        if PATTERNS[issue_name].search(text):
            issues.add(issue_name)

    if text_len < 20:
        issues.add('very_short')
    elif text_len > 50000:
        issues.add('very_long')
    return issues


def get_match_snippet(text, issue, snippet_chars=200):
    if issue in PATTERNS:
        m = PATTERNS[issue].search(text)
        if m:
            start = max(0, m.start() - 50)
            end = min(len(text), m.end() + 50)
            snippet = text[start:end]
            return {
                'match_text': m.group(),
                'match_span': [m.start(), m.end()],
                'context': ('...' + snippet + '...') if start > 0 else snippet[:snippet_chars]
            }
    if issue == 'very_short':
        return {'match_text': text, 'context': text}
    if issue == 'very_long':
        return {'match_text': '(' + str(len(text)) + ' chars)', 'context': text[:200] + '...'}
    return None


def audit_comprehensive(input_path, output_dir, sample_every=1, max_examples_per_category=200):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_records = 0
    scanned_records = 0
    empty_records = 0
    parse_errors = 0
    issue_counts = Counter()
    lang_issue_counts = defaultdict(Counter)
    multi_issue_records = []
    category_writers = {}
    category_counts = Counter()

    t0 = time.time()
    print("[audit] scanning " + str(input_path))
    print("[audit] sample_every=" + str(sample_every))
    print("[audit] max examples per category: " + str(max_examples_per_category))

    with Path(input_path).open('r') as f:
        for line in f:
            total_records += 1
            if total_records % sample_every != 0:
                continue
            scanned_records += 1

            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                parse_errors += 1
                continue

            text = rec.get('text', '')
            lang = rec.get('language', 'unknown')

            if not text or not text.strip():
                empty_records += 1
                continue

            issues = classify_record(text)
            if not issues:
                continue

            for issue in issues:
                issue_counts[issue] += 1
                lang_issue_counts[lang][issue] += 1

                if category_counts[issue] < max_examples_per_category:
                    if issue not in category_writers:
                        category_writers[issue] = (
                            output_dir / ('contaminated_' + issue + '.jsonl')
                        ).open('w')

                    snippet = get_match_snippet(text, issue)
                    example = {
                        'record_index': total_records - 1,
                        'language': lang,
                        'text_length': len(text),
                        'all_issues_in_record': sorted(issues),
                        'issue': issue,
                        'match_details': snippet,
                    }
                    category_writers[issue].write(
                        json.dumps(example, ensure_ascii=False) + '\n'
                    )
                    category_counts[issue] += 1

            if len(issues) >= 2 and len(multi_issue_records) < 200:
                multi_issue_records.append({
                    'record_index': total_records - 1,
                    'language': lang,
                    'text_length': len(text),
                    'issues': sorted(issues),
                    'text_preview': text[:300],
                })

            if scanned_records % 100000 == 0:
                elapsed = time.time() - t0
                rate = scanned_records / elapsed
                print("  scanned " + format(scanned_records, ',') + " records ("
                      + str(int(rate)) + " records/sec, issues so far: "
                      + format(sum(issue_counts.values()), ',') + ")")

    for w in category_writers.values():
        w.close()

    if multi_issue_records:
        with (output_dir / 'contaminated_multi_issue.jsonl').open('w') as f:
            for rec in multi_issue_records:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    total_scanned_nonempty = scanned_records - empty_records - parse_errors
    summary = {
        'input_file': str(input_path),
        'total_records_in_file': total_records,
        'scanned_records': scanned_records,
        'sampling': 'every ' + str(sample_every) + 'th',
        'sampling_coverage_pct': round(scanned_records / total_records * 100, 2),
        'empty_records': empty_records,
        'parse_errors': parse_errors,
        'nonempty_scanned': total_scanned_nonempty,
        'wall_clock_sec': round(time.time() - t0, 1),
        'issue_counts_absolute': dict(issue_counts),
        'issue_rates_pct': {
            issue: round(count / max(total_scanned_nonempty, 1) * 100, 4)
            for issue, count in issue_counts.items()
        },
        'estimated_full_corpus_counts': {
            issue: int(count * (total_records / max(scanned_records, 1)))
            for issue, count in issue_counts.items()
        },
        'per_language': {
            lang: dict(counts) for lang, counts in lang_issue_counts.items()
        },
        'multi_issue_records_count': len(multi_issue_records),
    }

    with (output_dir / 'summary.json').open('w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE (" + str(summary['wall_clock_sec']) + "s)")
    print("=" * 70)
    print("Total records:        " + format(total_records, ','))
    print("Scanned:              " + format(scanned_records, ',')
          + " (" + str(summary['sampling_coverage_pct']) + "% coverage)")
    print("  Empty:              " + format(empty_records, ','))
    print("  Parse errors:       " + format(parse_errors, ','))
    print("  Non-empty scanned:  " + format(total_scanned_nonempty, ','))

    print("\nIssue counts (in scanned):")
    if not issue_counts:
        print("  (no issues detected)")
    else:
        for issue, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
            rate = summary['issue_rates_pct'][issue]
            est = summary['estimated_full_corpus_counts'][issue]
            print("  " + issue.ljust(25) + " "
                  + format(count, ',').rjust(8) + "   ("
                  + format(rate, '6.4f') + "%)   ~"
                  + format(est, ',') + " in full corpus")

    print("\nPer-language breakdown:")
    for lang, counts in sorted(lang_issue_counts.items()):
        total_lang = sum(counts.values())
        if total_lang > 0:
            print("  " + lang + ": " + str(total_lang) + " total issues")
            for issue, count in sorted(counts.items(), key=lambda x: -x[1]):
                print("    " + issue.ljust(25) + " " + format(count, ',').rjust(6))

    print("\nMulti-issue records: " + format(len(multi_issue_records), ','))

    print("\nOutput files:")
    for f in sorted(output_dir.iterdir()):
        size = f.stat().st_size
        size_str = format(size, ',') + " bytes" if size < 1e6 else format(size / 1e6, '.1f') + " MB"
        print("  " + f.name.ljust(40) + " (" + size_str + ")")

    print("\n[write] summary at " + str(output_dir / 'summary.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--sample-every', type=int, default=1)
    parser.add_argument('--max-examples-per-category', type=int, default=200)
    args = parser.parse_args()
    audit_comprehensive(args.input, args.output_dir, args.sample_every, args.max_examples_per_category)
