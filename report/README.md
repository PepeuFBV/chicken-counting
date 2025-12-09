Report directory

This folder contains the paper skeleton and resources for writing the project report.

What I added:
- `paper.md` : a Markdown-first skeleton you can edit quickly.
- `main.tex` : a minimal LaTeX skeleton (if you prefer LaTeX).
- `references.bib` : bibliography file for BibTeX.
- `figs/` : a suggested folder for figures.

How to use:
- If you prefer Markdown, edit `paper.md` and convert to PDF with Pandoc:

```bash
pandoc report/paper.md -o report/paper.pdf --from markdown+yaml_metadata_block --pdf-engine=pdflatex --bibliography=report/references.bib
```

- If you prefer LaTeX, edit `main.tex` and compile with `pdflatex`/`bibtex` or `latexmk`:

```bash
cd report
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

If you want, I can fill sections using your repository content (methods, experiments, results).

----
Tips:
- Start with an outline and write the Methods/Experiments first — they're the easiest to make factual.
- Keep figures and tables in `figs/` and reference them from the text.
- Use `references.bib` for citations and keep entries minimal while drafting.
