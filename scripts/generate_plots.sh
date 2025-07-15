#!/bin/bash
git submodule init
git submodule update
./scripts/download_tensors.sh
./scripts/build_sparta.sh
./scripts/build_fastcc.sh
