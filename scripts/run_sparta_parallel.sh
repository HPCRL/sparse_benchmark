#!/bin/bash

export CURRENT_DIR=$(pwd)
export EXPERIMENT_MODES=3
export TENSOR_DIR=$CURRENT_DIR/fastcc_test_tensors
export SPARTA_DIR=$CURRENT_DIR/HiParTI

echo "SPARTA DIR is $SPARTA_DIR"

NUM_THREADS=$(nproc)

echo "Tensor: Chicago-0"
numactl --interleave=all --physcpubind=all $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/frostt/chicago-crime-comm.tns -Y $TENSOR_DIR/frostt/chicago-crime-comm.tns -m 1 -x 0 -y 0 -t $NUM_THREADS

echo "Tensor: Chicago-01"
numactl --interleave=all --physcpubind=all $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/frostt/chicago-crime-comm.tns -Y $TENSOR_DIR/frostt/chicago-crime-comm.tns -m 2 -x 0 1 -y 0 1 -t $NUM_THREADS

echo "Tensor: Chicago-123"
numactl --interleave=all --physcpubind=all $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/frostt/chicago-crime-comm.tns -Y $TENSOR_DIR/frostt/chicago-crime-comm.tns -m 3 -x 1 2 3 -y 1 2 3 -t $NUM_THREADS

echo "Tensor: Vast-5d-01"
numactl --interleave=all --physcpubind=all $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/frostt/vast-2015-mc1-5d.tns -Y $TENSOR_DIR/frostt/vast-2015-mc1-5d.tns -m 2 -x 0 1 -y 0 1 -t $NUM_THREADS
echo "Tensor: Vast-5d-014"
numactl --interleave=all --physcpubind=all $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/frostt/vast-2015-mc1-5d.tns -Y $TENSOR_DIR/frostt/vast-2015-mc1-5d.tns -m 3 -x 0 1 4 -y 0 1 4 -t $NUM_THREADS

echo "Tensor: Uber-02"
numactl --interleave=all --physcpubind=all $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/frostt/uber.tns -Y $TENSOR_DIR/frostt/uber.tns -m 2 -x 0 2 -y 0 2 -t $NUM_THREADS
echo "Tensor: Uber-123"
numactl --interleave=all --physcpubind=all $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/frostt/uber.tns -Y $TENSOR_DIR/frostt/uber.tns -m 3 -x 1 2 3 -y 1 2 3 -t $NUM_THREADS

echo "Tensor: NIPS-2"
numactl --interleave=all --physcpubind=all $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/frostt/nips.tns -Y $TENSOR_DIR/frostt/nips.tns -m 1 -x 2 -y 2 -t $NUM_THREADS

echo "Tensor: NIPS-23"
numactl --interleave=all --physcpubind=all $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/frostt/nips.tns -Y $TENSOR_DIR/frostt/nips.tns -m 2 -x 2 3 -y 2 3 -t $NUM_THREADS

echo "Tensor: NIPS-013"
numactl --interleave=all --physcpubind=all $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/frostt/nips.tns -Y $TENSOR_DIR/frostt/nips.tns -m 3 -x 0 1 3 -y 0 1 3 -t $NUM_THREADS

#caffeine
echo "Tensor: caffeine-vvoo"
numactl --interleave=all --physcpubind=all $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/caffeine/TEvv.tns -Y $TENSOR_DIR/caffeine/TEoo.tns -m 1 -x 2 -y 2 -t $NUM_THREADS

echo "Tensor: caffeine-ovov"
numactl --interleave=all --physcpubind=all $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/caffeine/TEov.tns -Y $TENSOR_DIR/caffeine/TEov.tns -m 1 -x 2 -y 2 -t $NUM_THREADS

echo "Tensor: caffeine-vvov"
numactl --interleave=all --physcpubind=all $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/caffeine/TEvv.tns -Y $TENSOR_DIR/caffeine/TEov.tns -m 1 -x 2 -y 2 -t $NUM_THREADS

#guanine
echo "Tensor: guanine-vvoo"
numactl --interleave=all --physcpubind=all $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/guanine/TEvv.tns -Y $TENSOR_DIR/guanine/TEoo.tns -m 1 -x 2 -y 2 -t $NUM_THREADS
echo "Tensor: guanine-ovov"
numactl --interleave=all --physcpubind=all $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/guanine/TEov.tns -Y $TENSOR_DIR/guanine/TEov.tns -m 1 -x 2 -y 2 -t $NUM_THREADS
echo "Tensor: guanine-vvov"
numactl --interleave=all --physcpubind=all $SPARTA_DIR/build/benchmark/ttt -X $TENSOR_DIR/guanine/TEvv.tns -Y $TENSOR_DIR/guanine/TEov.tns -m 1 -x 2 -y 2 -t $NUM_THREADS