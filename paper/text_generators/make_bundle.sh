#!/bin/bash
# Final packaging: rebuild the arXiv and EJOR PDFs and zip the sources.
set -e
cd /home/user/paper_revision
/home/user/venv/bin/python make_ejor.py > /dev/null
cd src
for main in main_arxiv main_ejor; do
  pdflatex -interaction=nonstopmode $main.tex > /dev/null; bibtex $main > /dev/null || true
  pdflatex -interaction=nonstopmode $main.tex > /dev/null; pdflatex -interaction=nonstopmode $main.tex > build_$main.log 2>&1
  echo "$main: $(grep -c '^!' build_$main.log) errors, $(grep -c -i 'undefined' build_$main.log) undefined, $(grep 'Output written' build_$main.log)"
done
pdflatex -interaction=nonstopmode highlights.tex > /dev/null 2>&1 || true
cd ..
STAMP=$(date +%Y%m%d_%H%M)
rm -rf bundle && mkdir -p bundle/falcom_sources
cp src/main_arxiv.tex src/main_ejor.tex src/highlights.tex src/ref_final.bib bundle/falcom_sources/
cp -r src/figures src/tex bundle/falcom_sources/ 2>/dev/null || true
cp src/main_arxiv.pdf bundle/falcom_arxiv_$STAMP.pdf; cp src/main_ejor.pdf bundle/falcom_ejor_$STAMP.pdf; cp src/highlights.pdf bundle/ 2>/dev/null || true
cd bundle && zip -qr falcom_sources_$STAMP.zip falcom_sources && ls -la
