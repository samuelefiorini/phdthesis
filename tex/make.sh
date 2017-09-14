#!/bin/bash
pdflatex -output-directory=./_build main.tex
bibtex main.aux
pdflatex -output-directory=./_build main.tex
pdflatex -output-directory=./_build main.tex
