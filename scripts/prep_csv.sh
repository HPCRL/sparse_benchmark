#!/bin/bash

mkdir data
cd data
mkdir molecules
cd ..

cat sparta_parallel.txt | ./scripts/extract_sparta.sh > sparta_parallel_full.csv

#and write to split files.

./scripts/split_sparta.sh sparta_parallel_full.csv
# mv sparta_frostt.csv sparta_parallel_frost.csv
# mv sparta_molecules.csv sparta_parallel_molecules.csv

./scripts/calculate_speedup.sh sparta_molecules.csv chemistry_times.csv > data/molecules/server_best.csv

./scripts/calculate_speedup sparta_frostt.csv frostt_times.csv > data/speedup_best_server.txt