#!/bin/bash

CURRENT_DIR=$(pwd)
export EXPERIMENT_MODES=3
export TENSOR_DIR=$CURRENT_DIR/fastcc_tensors
export SPARTA_DIR=$CURRENT_DIR/HiParTI

cp -f patches/ttt.c HiParTI/benchmark/ttt.c

cd HiParTI
mkdir build
cd build
cmake ..  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
make -j
cd ..
cd ..

