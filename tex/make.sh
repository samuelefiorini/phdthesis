#!/bin/bash

# trap ctrl-c and call ctrl_c()
trap ctrl_c SIGINT

function ctrl_c() {
        echo "** LaTeX compilation stopped... **"
	exit -1
}

mkdir ./_build
pdflatex -shell-escape -file-line-error -halt-on-error -output-directory=./_build main.tex
bibtex ./_build/main.aux

for i in `seq 1 2`; do
	pdflatex -shell-escape -file-line-error -halt-on-error -output-directory=./_build main.tex > /dev/null
        echo -n "."
done
echo

