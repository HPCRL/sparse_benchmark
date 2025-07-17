#!/bin/bash
git submodule init
git submodule update
#./scripts/download_tensors.sh
./scripts/build_sparta.sh
./scripts/build_fastcc.sh
./scripts/run_sparta_parallel.sh | tee sparta_parallel.txt
