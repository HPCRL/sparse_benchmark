#!/bin/bash

CURRENT_DIR=$(pwd)
export EXPERIMENT_MODES=3
export TENSOR_DIR=$CURRENT_DIR/fastcc_tensors
export SPARTA_DIR=$CURRENT_DIR/HiParTI

cd HiParTI
mkdir build -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cd build
cmake ..

