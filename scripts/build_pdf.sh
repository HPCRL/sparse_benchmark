#!/bin/bash
mv data latex/FastCC_AE

cd latex/FastCC_AE

pdflatex main.tex
pdflatex main.tex

cd ../..