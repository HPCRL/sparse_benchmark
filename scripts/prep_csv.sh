#!/bin/bash

mkdir data
cd data
mkdir molecules
cd ..

cd results
cat sparta_parallel.txt | ../scripts/extract_sparta.sh > sparta_parallel_full.csv

#and write to split files.

../scripts/split_fastcc_best_model.sh frostt_times.csv
../scripts/split_fastcc_best_model.sh chemistry_times.csv

../scripts/calculate_speedup.sh sparta_molecules.csv chemistry_times_best.csv > ../data/molecules/server_best.csv

../scripts/calculate_speedup sparta_frostt.csv frostt_times_best.csv > ../data/speedup_best_server.txt

../scripts/calculate_speedup.sh sparta_molecules.csv chemistry_times_model.csv > ../data/molecules/server_model.csv

../scripts/calculate_speedup sparta_frostt.csv frostt_times_model.csv > ../data/speedup_model_server.txt




cd ..