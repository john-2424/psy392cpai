# Beamer Deck -- Build Instructions

## Files

- `slides.tex` -- 14 content frames + 2 backup frames (adaptation grid, Lehnert ablation).
- `slides_notes.md` -- speaker notes, one block per slide. ~10 min at normal pace.

Figures are pulled from `../results/figures/` via `\graphicspath`.

## Build

```
cd project/slides
pdflatex slides.tex
pdflatex slides.tex
```

Output: `slides.pdf`. Double-compile ensures Beamer's navigation symbols / frame references resolve.

## Theme note

The deck uses `\usetheme{metropolis}` (the `mtheme` / `metropolis` package, a clean modern Beamer theme). If it is not installed:

- **MiKTeX**: the first `pdflatex` run will prompt to install it; accept.
- **TeX Live**: run `tlmgr install beamertheme-metropolis`.
- **No network / no install allowed**: in `slides.tex`, comment out `\usetheme{metropolis}` and uncomment the two-line fallback block right below it (`\usetheme{default}` + `\usecolortheme{dove}`). The deck will compile with the default Beamer look.

## Rehearsal checklist

1. `pdflatex slides.tex` twice; open `slides.pdf`. Confirm no frame overflows.
2. Read `slides_notes.md` aloud at presentation pace. Should land $\approx 10$ min. If over, trim slide 5, 6, or 13.
3. Verify every figure is legible at projector resolution (recent screens: fine at 1080p).

## Hooks for customization

- Change title-block affiliation / date / name at the top of `slides.tex`.
- Aspect ratio is `16:9` (`aspectratio=169`); switch to the default by removing that option in `\documentclass`.
- Add or drop a backup slide under `\appendix` without changing the 14-content-slide count.
