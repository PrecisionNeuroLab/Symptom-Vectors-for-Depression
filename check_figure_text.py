"""
Block TMLR figures that embed text not already public in this repo.

The interactive figures carry excerpts in their hover text. Every excerpt must come from a
corpus this repo already publishes (code/core_clinical_short, code/happydb, ...); withheld
corpora (books, redsm5) must show only the "text withheld" note. This finds every prose string
in the figure HTML and fails if one is not found, verbatim, in a tracked file outside the TMLR
folder.

    python check_figure_text.py          check the figure HTML staged for commit (the pre-commit hook)
    python check_figure_text.py --all    check every figure HTML in the working tree (CI)

Enable the hook once per clone with `git config core.hooksPath .githooks`.
"""
import argparse
import json
import re
import subprocess
import sys

KIT = 'tmlr-beyond-pdf-author-kit/'
FIGURES = KIT + 'submission_folder/'
MIN_WORDS = 8         # shorter strings are labels, not excerpts
MIN_LETTERS = 0.85    # share of letters among non-space characters; stats labels fall well below
MAX_REF_BYTES = 2_000_000
CSS = re.compile(r'\b(font-family|font-size|margin|padding|border(-radius)?|background|max-width)\s*:')


def git(*args, binary=False):
    out = subprocess.run(['git', *args], check=True, capture_output=True).stdout
    return out if binary else out.decode()


def normalize(text):
    return ' '.join(text.replace('<br>', ' ').split())


def reference_text():
    """Every tracked text file outside the TMLR folder, whitespace-normalized, in one string."""
    parts = []
    for path in git('ls-files', '-z').split('\0'):
        if not path or path.startswith(KIT) or path.endswith('.html'):
            continue
        try:
            with open(path, 'rb') as f:
                data = f.read(MAX_REF_BYTES + 1)
        except OSError:
            continue
        if len(data) > MAX_REF_BYTES or b'\0' in data[:8192]:
            continue  # binary (tensors) or too large to be a corpus text
        try:
            parts.append(normalize(data.decode('utf-8')))
        except UnicodeDecodeError:
            continue
    return '\n'.join(parts)


def prose(html):
    """Distinct prose strings embedded in a figure's JSON (hover text, labels)."""
    found = set()
    for raw in re.findall(r'"((?:[^"\\]|\\.){40,})"', html):
        try:
            text = normalize(json.loads(f'"{raw}"'))
        except json.JSONDecodeError:
            continue
        if re.search(r'[<>{}]', text) or CSS.search(text):
            continue  # markup, templates, inline styles
        chars = [c for c in text if not c.isspace()]
        letters = sum(c.isalpha() or c in ".,;:'’\"“”-–—()!?" for c in chars)
        if len(text.split()) >= MIN_WORDS and letters / len(chars) >= MIN_LETTERS:
            found.add(text)
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--all', action='store_true', help='check every figure in the working tree')
    args = parser.parse_args()

    if args.all:
        files = [p for p in git('ls-files', FIGURES).splitlines() if p.endswith('.html')]
        read = lambda p: open(p, encoding='utf-8').read()
    else:
        files = [p for p in git('diff', '--cached', '--name-only', '--diff-filter=ACMR', '--', FIGURES).splitlines()
                 if p.endswith('.html')]
        read = lambda p: git('show', f':{p}')  # the staged version, which is what gets committed
    if not files:
        return 0

    reference = reference_text()
    bad = {}
    n = 0
    for path in files:
        strings = prose(read(path))
        n += len(strings)
        missing = sorted(s for s in strings if s not in reference)
        if missing:
            bad[path] = missing

    if not bad:
        print(f'check_figure_text: {n} text passages in {len(files)} figure(s), all already public')
        return 0
    print('check_figure_text: these figures embed text that is not public in this repo:', file=sys.stderr)
    for path, missing in bad.items():
        print(f'\n  {path}', file=sys.stderr)
        for s in missing[:10]:
            print(f'    - {s[:160]}', file=sys.stderr)
        if len(missing) > 10:
            print(f'    ... and {len(missing) - 10} more', file=sys.stderr)
    print('\nWithhold them in the figure (as for books / redsm5), or publish the source text first.',
          file=sys.stderr)
    return 1


if __name__ == '__main__':
    sys.exit(main())
