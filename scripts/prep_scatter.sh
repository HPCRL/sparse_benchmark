#!/bin/bash

mkdir data
cd data
mkdir molecules
cd ..

#generate tile sets for graphs.
cat fastcc_tiles.txt | ./scripts/extract_tiles.sh | ./scripts/split_fastcc_tiles.sh

#generates split_molecules.csv and split_frostt.csv

cat split_frostt.csv  | ./scripts/convert_tiles_to_csv.sh > data/main_scatter.txt

cat split_molecules.csv | ./scripts/convert_tiles_to_csv.sh > data/molecule_scatter.txt
