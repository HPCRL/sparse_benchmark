#!/bin/bash

rm -r latex/FastCC_AE/data
cp -R data latex/FastCC_AE

cd latex/FastCC_AE

pdflatex main.tex
pdflatex main.tex

mv main.pdf ../../
cd ../..