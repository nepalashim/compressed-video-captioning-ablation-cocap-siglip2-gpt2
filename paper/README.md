# Preprint source

- `main.tex` - the paper. Every unfilled item is wrapped in `\TODO{...}`; provisional numbers use
  `\num{...}`. Both render red, so nothing unfinished can reach a PDF unnoticed.
- `figures/architecture.tex` - TikZ architecture diagram, `\input` into `main.tex`. Self-contained,
  so the arXiv source needs no binary assets.
- `references.bib`
- `NOTES.md` - **read this before writing any results section.** Running record of measurements,
  deviations and caveats that must appear in the paper.

## Build

```bash
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Before submitting

```bash
grep -n "TODO" main.tex        # must return nothing
grep -n "num{" main.tex        # every provisional number re-checked against the final run
```
