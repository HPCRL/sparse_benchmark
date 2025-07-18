#!/bin/bash
git submodule init
git submodule update
./scripts/download_tensors.sh
./scripts/build_sparta.sh
./scripts/build_fastcc.sh
./scripts/run_fastcc.sh
./scripts/run_sparta_parallel.sh | tee results/sparta_parallel.txt
./scripts/prep_scatter.sh
./scripts/prep_csv.sh
./scripts/build_pdf.sh