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
- **Orphan check (only when over length).** This one is not a standing rule.
  Run it when the paper sits slightly over a page limit, where reclaiming
  short final lines is the cheapest way to recover space. Do not spend edits
  on it otherwise.
  `python paper/check_orphans.py` reports paragraphs ending in a line of one
  or two words. It reads the typeset PDF, since whether a paragraph ends short
  depends on the layout. Fix by shortening or rephrasing the sentence that
  ends there, never by padding. It skips run-in headings and the reference
  list, but table rows can still show up as false positives, so read the
  output rather than trusting the count.

## Antal's voice

Calibrated from his 2026-07-31 pass over the paper. Draft this way from the
start rather than making him strip it out.

- **Cut meta-commentary.** Sentences about what the paper is doing get
  deleted. "We make both routes explicit here, as together they are central
  to..." and "This is the mechanism behind the saturation reported in..."
  both went. State the thing; do not announce that you are stating it.
- **No self-praise adjectives.** "first-class mechanism" became "mechanism";
  "and crucially they span" became "spanning"; "an honest account of what
  that buys" was cut. Avoid "crucially", "importantly", "notably".
- **Never call your own work honest.** "under honest evaluation" became
  "under fair evaluation". Use "fair", "budget-matched", "controlled".
- **Plain verbs over vivid ones.** "rolled out greedily" became "expanded
  greedily"; "units that are simply never exercised" became "units that are
  not used"; "pins the budget" became "reaches the budget". A sentence of
  mine built on "the constraint bites in practice" was cut outright.
- **No anthropomorphising.** "it decides who is at risk" became "which units
  are at risk".
- **Attribute claims to their authors.** "behavioural evidence that multi-word
  units are psycholinguistically real" became "which they take as behavioural
  evidence that...". Do not assert someone else's interpretation as fact.
- **Moderate rhetorical labels in formal claims.** In the contributions list
  "forgetting is load-bearing" became "passive forgetting is functional",
  even though the thesis keeps the stronger phrase elsewhere.
- **Prefer agentive first person.** "This paper asks" became "Here, we ask".
- **Split long coordinated sentences.** A comma-plus-"and" joining two full
  clauses tends to become a full stop.
- **Define terms concretely.** "Symbol: character baseline." became "Symbol: a
  baseline tokenizer that segments every single character."
- He comments lines out with `%` rather than deleting them. Leave his
  commented text alone; it is a decision in progress, not dead weight.

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
