#include "sparse_opcount/contract.hpp"
#include "sparse_opcount/read.hpp"
#include <chrono>
#include <iostream>
#include <sys/resource.h>
#include <sys/wait.h>
#include <vector>
static float scaling_factor = 1;

void make_a_run(Tensor<double>& some_tensor, std::string exp_name, CoOrdinate contr, int left_tile_size, int right_tile_size, bool dense, int lt, int rt, std::ofstream& out){
    if (fork() == 0) {
        some_tensor._infer_dimensionality();
        some_tensor._infer_shape();
        std::chrono::high_resolution_clock::time_point t1 =
            std::chrono::high_resolution_clock::now();
        if(dense){
            ListTensor<double> result = some_tensor.parallel_tile2d_outer_multiply<TileAccumulator<double>, double>(
                some_tensor, contr, contr, left_tile_size, right_tile_size);
        } else {
            ListTensor<double> result = some_tensor.parallel_tile2d_outer_multiply<TileAccumulatorMap<double>, double>(
                some_tensor, contr, contr, left_tile_size, right_tile_size);
        }
        std::chrono::high_resolution_clock::time_point t2 =
            std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span =
            std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
        out<<exp_name<<","<<lt<<","<<rt<<","<<(dense?"dense":"sparse")<<","<<time_span.count()<<std::endl;
        exit(0);
    } else {
        int stat;
        wait(&stat);
    }
}

void run_frostt_experiments(int left_tile_size, int right_tile_size, std::ofstream& out) {
  std::string prefix = "/media/saurabh/New Volume1/ubuntu_downloads/frostt/";
 
  // nips experiments
  std::cout << "Running nips tensor" << std::endl;
  Tensor<double> nips = Tensor<double>(prefix + "/nips.tns", true);
  make_a_run(nips, "NIPS 2", CoOrdinate({2}), left_tile_size, right_tile_size, true, left_tile_size, right_tile_size, out);
  make_a_run(nips, "NIPS 2", CoOrdinate({2}), left_tile_size, right_tile_size, false, left_tile_size, right_tile_size, out);
  make_a_run(nips, "NIPS 2 3", CoOrdinate({2, 3}), left_tile_size, right_tile_size, true, left_tile_size, right_tile_size, out);
  make_a_run(nips, "NIPS 2 3", CoOrdinate({2, 3}), left_tile_size, right_tile_size, false, left_tile_size, right_tile_size, out);
  make_a_run(nips, "NIPS 0 1 3", CoOrdinate({0, 1, 3}), left_tile_size, right_tile_size, true, left_tile_size, right_tile_size, out);
  make_a_run(nips, "NIPS 0 1 3", CoOrdinate({0, 1, 3}), left_tile_size, right_tile_size, false, left_tile_size, right_tile_size, out);

  ////////// chicago experiments
  std::cout << "Running chicago tensor" << std::endl;
  Tensor<double> chicago = Tensor<double>(prefix + "/chicago-crime-comm.tns", true);
  make_a_run(chicago, "Chicago 0", CoOrdinate({0}), left_tile_size, right_tile_size, true, left_tile_size, right_tile_size, out);
  make_a_run(chicago, "Chicago 0", CoOrdinate({0}), left_tile_size, right_tile_size, false, left_tile_size, right_tile_size, out);
  make_a_run(chicago, "Chicago 0 1", CoOrdinate({0, 1}), left_tile_size, right_tile_size, true, left_tile_size, right_tile_size, out);
  make_a_run(chicago, "Chicago 0 1", CoOrdinate({0, 1}), left_tile_size, right_tile_size, false, left_tile_size, right_tile_size, out);
  make_a_run(chicago, "Chicago 1 2 3", CoOrdinate({1, 2, 3}), left_tile_size, right_tile_size, true, left_tile_size, right_tile_size, out);
  make_a_run(chicago, "Chicago 1 2 3", CoOrdinate({1, 2, 3}), left_tile_size, right_tile_size, false, left_tile_size, right_tile_size, out);
 
  //////////////// vast-3d experiments
  std::cout << "Running vast-5d tensor" << std::endl;
  Tensor<double> vast = Tensor<double>(prefix + "/vast-2015-mc1-5d.tns", true);
  make_a_run(vast, "Vast 0 1", CoOrdinate({0, 1}), left_tile_size, right_tile_size, true, left_tile_size, right_tile_size, out);
  make_a_run(vast, "Vast 0 1", CoOrdinate({0, 1}), left_tile_size, right_tile_size, false, left_tile_size, right_tile_size, out);
  make_a_run(vast, "Vast 0 1 4", CoOrdinate({0, 1, 4}), left_tile_size, right_tile_size, true, left_tile_size, right_tile_size, out);
  make_a_run(vast, "Vast 0 1 4", CoOrdinate({0, 1, 4}), left_tile_size, right_tile_size, false, left_tile_size, right_tile_size, out);

  /////////////////// uber experiments
  std::cout << "Running uber tensor" << std::endl;
  Tensor<double> uber = Tensor<double>(prefix + "/uber.tns", true);
  make_a_run(uber, "Uber 0 2", CoOrdinate({0, 2}), left_tile_size, right_tile_size, true, left_tile_size, right_tile_size, out);
  make_a_run(uber, "Uber 0 2", CoOrdinate({0, 2}), left_tile_size, right_tile_size, false, left_tile_size, right_tile_size, out);
  make_a_run(uber, "Uber 1 2 3", CoOrdinate({1, 2, 3}), left_tile_size, right_tile_size, true, left_tile_size, right_tile_size, out);
  make_a_run(uber, "Uber 1 2 3", CoOrdinate({1, 2, 3}), left_tile_size, right_tile_size, false, left_tile_size, right_tile_size, out);
}

int main() {
    std::ofstream file;
    file.open("frostt_tilesweep_contd.csv");
    file << "Tensor mode,left_tile_size, right_tile_size,acc_type,time" << std::endl;
    // seems like this ran till 512x512, scaling factor 8
    for(int base_dim = 512; base_dim <= 1024; base_dim *= 2){
        if (base_dim == 512) {
            for (int scaling = 16; scaling <= base_dim; scaling *= 2) {
              run_frostt_experiments(base_dim / scaling, base_dim * scaling,
                                     file);
            }
        } else {
            for (int scaling = 1; scaling <= base_dim; scaling *= 2) {
              run_frostt_experiments(base_dim / scaling, base_dim * scaling,
                                     file);
            }
        }
    }
    
    file.close();
}
