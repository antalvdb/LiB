# House rules

## Prose style

These apply to the paper and to anything else written for this project.

- **No italics for emphasis.** Do not use `\emph{}` or italics to stress a
  word. Rewrite so the sentence carries the stress itself, usually by putting
  the important term at the start or end of the clause.
  - Italics remain correct for their conventional non-emphatic uses:
    linguistic examples and cited forms (*of the*, *Nord- und Südamerika*),
    foreign-language terms (*doğum*, *nisba*), and titles.
- **Avoid the em-dash.** Use a comma, a colon, parentheses, or a full stop
  instead. A colon works when the second half explains the first; a full stop
  works when the aside is really a separate thought.

## Paper

- Target venue: Language Resources and Evaluation (Springer journal 10579).
  Class `sn-jnl` with the `sn-apa` reference style, APA 7 author-year.
- Springer requires a single `.tex` file. `paper/forgetting_section.tex` was
  inlined into `paper/paper.tex` and is no longer included by anything.
- Full-length papers run 18 to 25 pages; the abstract must be 150 to 250
  words, and there must be 4 to 6 keywords.
- The paper is built with pdfLaTeX and has no font support for non-Latin
  script. Arabic and Chinese material must be transliterated or described.
- `lib-paper-overleaf.zip` is the upload bundle: `paper.tex`,
  `references.bib`, `sn-jnl.cls`, `sn-apacite.bst`. Rebuild it after editing.
- Never put numbers in the paper that came from a run predating the
  forgetting fix. Date a run by the build of the compiled extension,
  2026-07-23 17:32, not by the git commit of the fix.

## Git

- The GPG key expired on 2026-05-01, so `commit.gpgsign=true` makes every
  commit fail on the first attempt. Commit with `--no-gpg-sign`, which matches
  every commit since May.
