# Clean the folder
rm -rf _build
rm *.aux
rm *.log
rm *.out
rm *.toc
rm *.fls
rm *.brf
rm *.synctex.gz
rm *.bbl
rm *.blg
rm *.lof
rm *.lot
rm *.fdb_latexmk

# Prepare for the next build
mkdir _build
touch _build/main.pdf
ln -s _build/main.pdf main.pdf
