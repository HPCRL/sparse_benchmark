#include "sparse_opcount/contract.hpp"
#include "sparse_opcount/read.hpp"
#include <chrono>
#include <iostream>
#include <sys/resource.h>
#include <sys/wait.h>
#include <vector>

void make_a_run(Tensor<double>& some_tensor, std::string exp_name, CoOrdinate contr, int left_tile_size, int right_tile_size, bool dense, int lt, int rt, std::ostream& out){
    if (fork() == 0) {
        some_tensor._infer_dimensionality();
        some_tensor._infer_shape();
        std::chrono::high_resolution_clock::time_point t1 =
            std::chrono::high_resolution_clock::now();
        if(dense){
            ListTensor<double> result = some_tensor.fastcc_multiply<TileAccumulator<double>, double>(
                some_tensor, contr, contr, left_tile_size);
        } else {
            ListTensor<double> result = some_tensor.fastcc_multiply<TileAccumulatorMap<double>, double>(
                some_tensor, contr, contr, left_tile_size);
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

void run_a_times_b(Tensor<double>& a, Tensor<double>& b, std::string exp_name, CoOrdinate contr, int tile_size, bool dense, std::ostream& out){
    if (fork() == 0) {
        a._infer_dimensionality();
        a._infer_shape();
        b._infer_dimensionality();
        b._infer_shape();
        std::chrono::high_resolution_clock::time_point t1 =
            std::chrono::high_resolution_clock::now();
        if(dense){
            ListTensor<double> result = a.fastcc_multiply<TileAccumulator<double>, double>(
                b, contr, contr, tile_size);
        } else {
            ListTensor<double> result = a.fastcc_multiply<TileAccumulatorMap<double>, double>(
                b, contr, contr, tile_size);
        }
        std::chrono::high_resolution_clock::time_point t2 =
            std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span =
            std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
        out<<exp_name<<","<<tile_size<<","<<(dense?"dense":"sparse")<<","<<time_span.count()<<std::endl;
        exit(0);
    } else {
        int stat;
        wait(&stat);
    }
}

void run_frostt_experiments(int left_tile_size, int right_tile_size, std::ostream& out, std::string& prefix) {

  // nips experiments
  std::cout << "Running nips tensor" << std::endl;
  Tensor<double> nips = Tensor<double>(prefix + "/nips.tns", true);
  make_a_run(nips, "NIPS 2", CoOrdinate({2}), left_tile_size, right_tile_size, false, left_tile_size, right_tile_size, out);
  make_a_run(nips, "NIPS 2 3", CoOrdinate({2, 3}), left_tile_size, right_tile_size, false, left_tile_size, right_tile_size, out);
  make_a_run(nips, "NIPS 0 1 3", CoOrdinate({0, 1, 3}), left_tile_size, right_tile_size, true, left_tile_size, right_tile_size, out);

  ////////////// chicago experiments
  std::cout << "Running chicago tensor" << std::endl;
  Tensor<double> chicago = Tensor<double>(prefix + "/chicago-crime-comm.tns", true);
  make_a_run(chicago, "Chicago 0", CoOrdinate({0}), left_tile_size, right_tile_size, true, left_tile_size, right_tile_size, out);
  make_a_run(chicago, "Chicago 0 1", CoOrdinate({0, 1}), left_tile_size, right_tile_size, true, left_tile_size, right_tile_size, out);
  make_a_run(chicago, "Chicago 1 2 3", CoOrdinate({1, 2, 3}), left_tile_size, right_tile_size, true, left_tile_size, right_tile_size, out);

  //////////////// vast-3d experiments
  std::cout << "Running vast-5d tensor" << std::endl;
  Tensor<double> vast = Tensor<double>(prefix + "/vast-2015-mc1-5d.tns", true);
  make_a_run(vast, "Vast 0 1", CoOrdinate({0, 1}), left_tile_size, right_tile_size, true, left_tile_size, right_tile_size, out);
  make_a_run(vast, "Vast 0 1 4", CoOrdinate({0, 1, 4}), left_tile_size, right_tile_size, true, left_tile_size, right_tile_size, out);

  /////////////////// uber experiments
  std::cout << "Running uber tensor" << std::endl;
  Tensor<double> uber = Tensor<double>(prefix + "/uber.tns", true);
  make_a_run(uber, "Uber 0 2", CoOrdinate({0, 2}), left_tile_size, right_tile_size, true, left_tile_size, right_tile_size, out);
  make_a_run(uber, "Uber 1 2 3", CoOrdinate({1, 2, 3}), left_tile_size, right_tile_size, true, left_tile_size, right_tile_size, out);
}

void run_chemistry_experiments(int tile_size, std::ostream& out, std::string& caffeine_prefix, std::string& guanine_prefix) {

  ///////////// caffeine experiments
  std::cout << "Running caffeine VVOO" << std::endl;
  Tensor<double> tevv = Tensor<double>(caffeine_prefix + "/TEvv.tns", true);
  Tensor<double> teoo = Tensor<double>(caffeine_prefix + "/TEoo.tns", true);
  run_a_times_b(tevv, teoo, "caffeine VVOO", CoOrdinate({2}), tile_size, true, out);
  Tensor<double> teov = Tensor<double>(caffeine_prefix + "/TEov.tns", true);
  run_a_times_b(teov, teov, "caffeine OVOV", CoOrdinate({2}), tile_size, true, out);
  run_a_times_b(tevv, teov, "caffeine VVOV", CoOrdinate({2}), tile_size, true, out);


  std::cout << "Running guanine VVOO" << std::endl;
  tevv = Tensor<double>(guanine_prefix + "/TEvv.tns", true);
  teoo = Tensor<double>(guanine_prefix + "/TEoo.tns", true);
  run_a_times_b(tevv, teoo, "guanine VVOO", CoOrdinate({2}), tile_size, true, out);
  teov = Tensor<double>(guanine_prefix + "/TEov.tns", true);
  run_a_times_b(teov, teov, "guanine OVOV", CoOrdinate({2}), tile_size, true, out);
  run_a_times_b(tevv, teov, "guanine VVOV", CoOrdinate({2}), tile_size, true, out);
}

int main(int argc, char** argv) {
    if(argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <<path to folder containing frostt tensors>> <<path to folder containing caffeine tensors>> <<path to folder containing guanine tensors>>" << std::endl;
        return 1;
    }
    std::string frostt_dir = argv[1], caffeine_dir = argv[2], guanine_dir = argv[3];
    std::ofstream results_chem, results_frostt;
    results_chem.open("chemistry_times.csv");
    results_chem << "Tensor mode,tile_size,acc_type,time" << std::endl;
    run_chemistry_experiments(512, results_chem, caffeine_dir, guanine_dir);
    results_chem.close();
    results_frostt.open("frostt_times.csv");
    results_frostt << "Tensor mode,tile_size,acc_type,time" << std::endl;
    run_frostt_experiments(512, 512, results_frostt, frostt_dir);
    results_frostt.close();
    return 0;
}
