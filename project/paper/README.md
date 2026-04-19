# NeurIPS 2024 Report -- Build Instructions

This directory contains the LaTeX source for the final report, written against the NeurIPS 2024 style.

## Files

- `main.tex` -- main paper body (abstract, intro, methods, results, discussion, conclusion).
- `appendix.tex` -- Appendix A (hyperparameters), B (per-seed adaptation metrics), C (per-seed probe), D (reproduce instructions). Included via `\input{appendix.tex}` at the end of `main.tex`.
- `references.bib` -- BibTeX bibliography (14 entries).
- `neurips_2024.sty` -- **MUST be downloaded into this directory before the first build.** Get it from <https://neurips.cc/Conferences/2024/PaperInformation/StyleFiles> (the `Style Files` link on the NeurIPS 2024 author page). License: free for academic use. It is deliberately not committed to the repo to avoid redistribution questions.

Figures are pulled from `../results/figures/` via `\graphicspath` -- do not copy them.

## Build

Requires a TeX distribution with `pdflatex` and `bibtex` on `PATH` (MiKTeX or TeX Live).

```
cd project/paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Output: `main.pdf`. The double-`pdflatex` after `bibtex` is required so that in-text `\cite{}` keys resolve correctly.

## Required LaTeX packages

All packages used are in standard LaTeX distributions (MiKTeX/TeX Live default install):
`inputenc, fontenc, lmodern, microtype, graphicx, booktabs, amsmath, amssymb, url, hyperref, xcolor, caption, subcaption, array, natbib` (pulled in by `neurips_2024.sty`).

If the first `pdflatex` run complains about a missing package, MiKTeX will offer to auto-install it; accept. For TeX Live, run `tlmgr install <package-name>`.

## Troubleshooting

- **`! LaTeX Error: File 'neurips_2024.sty' not found.`** -- you skipped the download step above.
- **`LaTeX Warning: Citation 'foo' on page X undefined`** -- run `bibtex main` then re-run `pdflatex main.tex` twice.
- **Figures missing** -- make sure you build from `project/paper/` (so `../results/figures/` resolves) and that the notebook has been executed at least once to populate the figures directory.
