# Compiling and submitting

## Overleaf

Upload `main.tex`, `references.bib` and the `figures/` folder (keep the folder structure - the
figure is pulled in by relative path). Set **`main.tex` as the main document**. Overleaf runs
`pdflatex -> bibtex -> pdflatex x2` automatically.

Everything used is stock TeX Live: `geometry`, `times`, `graphicx`, `amsmath`, `booktabs`,
`hyperref`, `tikz`, `xcolor`. No custom class or style file.

Leave `NOTES.md` and this file out - they are working notes, not part of the paper.

## Before submitting

```bash
grep -n "TODO" main.tex      # must return nothing
```

Unfilled items render as red `[TODO: ...]` boxes, so they are visible in the PDF as well.

Also check by eye:
- the architecture figure renders legibly and does not overflow its column
- Table 5 (qualitative captions) fits the two-column width
- no `??` in place of a citation or cross-reference (means bibtex did not run)

## arXiv

arXiv compiles the source itself; it does not accept a PDF built elsewhere for a LaTeX paper.
Upload a zip of:

```
main.tex
references.bib
main.bbl            <- required: arXiv does not run bibtex
figures/architecture.tex
```

Get `main.bbl` from Overleaf via *Logs and output files -> Other logs and files*. Without it the
bibliography comes out empty.

Suggested categories: **cs.CV** primary, **cs.CL** cross-list.

## Still open before it is submittable

- [ ] `\TODO{repository URL}` in the appendix - the only marker left
- [ ] confirm the author name and affiliation are right
- [ ] decide whether to make the GitHub repository public; the appendix promises code
