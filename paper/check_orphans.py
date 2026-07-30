"""Flag paragraphs whose last line holds only one or two words.

A justified paragraph sets every line to the full measure except the last, so
a very short final line leaves a conspicuous gap. The fix is always to shorten
or rephrase the sentence that ends there, pulling the stray words up.

Detection works on the typeset PDF, not the source, because whether a
paragraph ends short depends on the layout. A line is reported when it holds
at most `--max-words` words and the line before it runs at least
`--full-line` characters, which is what marks it as the tail of a real
paragraph rather than a stray fragment.

Two classes of false positive are filtered out. Run-in headings such as
`\\paragraph{Competing interests.}` sit on their own line in the extracted
text and look exactly like a short final line, so the headings are read out of
the .tex and skipped. Bibliography entries are set with a hanging indent and
routinely end short, and no rephrasing is available, so everything from the
reference list onward is ignored.

Usage
-----
    python paper/check_orphans.py                       # paper.pdf + paper.tex
    python paper/check_orphans.py --max-words 3
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

HEADING = re.compile(r"\\(?:sub)*section\*?\{([^}]*)\}|\\paragraph\{([^}]*)\}")
PAGE_NO = re.compile(r"^\s*\d+\s*$")
SEC_NO = re.compile(r"^\d+(\.\d+)*\s+")   # "3.4 Metrics" as typeset
STOP_AT = ("References",)


def strip_tex(s):
    s = re.sub(r"\\[a-zA-Z]+\s*", "", s)
    return re.sub(r"[{}$\\]", "", s).strip()


def headings_from_source(tex):
    """Heading strings that will appear on their own line in the PDF text."""
    if not tex or not Path(tex).exists():
        return set()
    src = Path(tex).read_text(encoding="utf-8")
    out = set()
    for m in HEADING.finditer(src):
        text = strip_tex(m.group(1) or m.group(2) or "")
        if text:
            out.add(text.rstrip("."))
    return out


def pdf_lines(pdf):
    out = subprocess.run(["pdftotext", str(pdf), "-"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"pdftotext failed: {out.stderr.strip()}")
    for pno, page in enumerate(out.stdout.split("\f"), 1):
        for line in page.split("\n"):
            yield pno, line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", nargs="?", default="paper/paper.pdf")
    ap.add_argument("--tex", default="paper/paper.tex",
                    help="source, read only to identify run-in headings")
    ap.add_argument("--max-words", type=int, default=2)
    ap.add_argument("--full-line", type=int, default=55)
    ap.add_argument("--skip-pages", type=int, default=1,
                    help="ignore this many leading pages (title page)")
    args = ap.parse_args()

    pdf = Path(args.pdf)
    if not pdf.exists():
        sys.exit(f"no such file: {pdf}")

    heads = headings_from_source(args.tex)
    lines = list(pdf_lines(pdf))

    hits, prev = [], ""
    for pno, raw in lines:
        text = raw.strip()
        if text.rstrip(".") in STOP_AT:
            break
        if not text or PAGE_NO.match(text):
            prev = text or prev
            continue
        bare = SEC_NO.sub("", text).rstrip(".")
        words = text.split()
        if (len(words) <= args.max_words
                and len(prev) >= args.full_line
                and pno > args.skip_pages
                and bare not in heads):
            hits.append((pno, text, prev))
        prev = text

    if not hits:
        print(f"{pdf}: clean, no paragraph ends in "
              f"{args.max_words} words or fewer")
        return 0

    print(f"{pdf}: {len(hits)} paragraph(s) ending in "
          f"{args.max_words} words or fewer\n")
    for pno, text, prev in hits:
        print(f"  p.{pno}: ...{prev[-62:]}")
        print(f"         >>> {text}\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
