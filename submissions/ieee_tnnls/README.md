# OrganoidEnv manuscript package (revised)

Contents: `manuscript.tex`, `references.bib`, `manuscript.bbl`, `IEEEtran.cls`, `manuscript.pdf`, `figures/` (PDF), `CHANGES.md`.

Build (TeX Live with IEEEtran.bst installed):

    pdflatex manuscript && bibtex manuscript && pdflatex manuscript && pdflatex manuscript

Items shown in red in the PDF (author block, funding, repository/DOI, license) are placeholders you must fill.
`% TODO(author)` comments in manuscript.tex mark numbers or descriptions to confirm against your logs.
This package contains no code, logs, or raw results; all numbers come from the tables in the manuscript.
