#!/bin/bash

export CURRENT_DIR=$(pwd)
export EXPERIMENT_MODES=3
export TENSOR_DIR=$CURRENT_DIR/fastcc_test_tensors
export SPARTA_DIR=$CURRENT_DIR/HiParTI

echo "SPARTA DIR is $SPARTA_DIR"

NUM_THREADS=$(nproc)

numactl --interleave=all --physcpubind=all $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/frostt/chicago-crime-comm.tns -Y $TENSOR_DIR/frostt/chicago-crime-comm.tns -m 1 -x 0 -y 0 -t $NUM_THREADS

numactl --interleave=all --physcpubind=all $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/frostt/chicago-crime-comm.tns -Y $TENSOR_DIR/frostt/chicago-crime-comm.tns -m 2 -x 0 1 -y 0 1 -t $NUM_THREADS

numactl --interleave=all --physcpubind=all $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/frostt/chicago-crime-comm.tns -Y $TENSOR_DIR/frostt/chicago-crime-comm.tns -m 3 -x 1 2 3 -y 1 2 3 -t $NUM_THREADS


numactl --interleave=all --physcpubind=all $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/frostt/vast-2015-mc1-5d.tns -Y $TENSOR_DIR/frostt/vast-2015-mc1-5d.tns -m 2 -x 0 1 -y 0 1 -t $NUM_THREADS

numactl --interleave=all --physcpubind=all $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/frostt/vast-2015-mc1-5d.tns -Y $TENSOR_DIR/frostt/vast-2015-mc1-5d.tns -m 3 -x 0 1 4 -y 0 1 4 -t $NUM_THREADS

numactl --interleave=all --physcpubind=all $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/frostt/uber.tns -Y $TENSOR_DIR/frostt/uber.tns -m 2 -x 0 2 -y 0 2 -t $NUM_THREADS

numactl --interleave=all --physcpubind=all $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/frostt/uber.tns -Y $TENSOR_DIR/frostt/uber.tns -m 3 -x 1 2 3 -y 1 2 3 -t $NUM_THREADS

numactl --interleave=all --physcpubind=all $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/frostt/nips.tns -Y $TENSOR_DIR/frostt/nips.tns -m 1 -x 2 -y 2 -t $NUM_THREADS

numactl --interleave=all --physcpubind=all $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/frostt/nips.tns -Y $TENSOR_DIR/frostt/nips.tns -m 2 -x 2 3 -y 2 3 -t $NUM_THREADS

numactl --interleave=all --physcpubind=all $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/frostt/nips.tns -Y $TENSOR_DIR/frostt/nips.tns -m 3 -x 0 1 3 -y 0 1 3 -t $NUM_THREADS

#caffeine
numactl --interleave=all --physcpubind=all $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/caffeine/TEov.tns -Y $TENSOR_DIR/caffeine/TEov.tns -m 1 -x 2 -y 2 -t $NUM_THREADS

numactl --interleave=all --physcpubind=all $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/caffeine/TEvv.tns -Y $TENSOR_DIR/caffeine/TEoo.tns -m 1 -x 2 -y 2 -t $NUM_THREADS

numactl --interleave=all --physcpubind=all $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/caffeine/TEvv.tns -Y $TENSOR_DIR/caffeine/TEov.tns -m 1 -x 2 -y 2 -t $NUM_THREADS

#guanine
numactl --interleave=all --physcpubind=all $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/guanine/TEov.tns -Y $TENSOR_DIR/guanine/TEov.tns -m 1 -x 2 -y 2 -t $NUM_THREADS

numactl --interleave=all --physcpubind=all $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/guanine/TEvv.tns -Y $TENSOR_DIR/guanine/TEoo.tns -m 1 -x 2 -y 2 -t $NUM_THREADS

numactl --interleave=all --physcpubind=all $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/guanine/TEvv.tns -Y $TENSOR_DIR/guanine/TEov.tns -m 1 -x 2 -y 2 -t $NUM_THREADS