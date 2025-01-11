#include "sparse_opcount/contract.hpp"
#include "sparse_opcount/read.hpp"
#include <chrono>
#include <iostream>
#include <sys/resource.h>
#include <sys/wait.h>
#include <vector>
static float scaling_factor = 25;


// void test_teov_pao_contraction(Tensor<double> teov){
//     int MO = 3;
//     int PAO = 23;
//     int AUX = 374;
//     tsl::hopscotch_set<CoOrdinate> output_nnzs;
//     for (int i = 0; i < MO; i++) {
//       for (int j = 0; j < MO; j++) {
//         for (int e_mu = 0; e_mu < PAO; e_mu++) {
//           for (int f_mu = 0; f_mu < PAO; f_mu++) {
//             double sum = 0.0;
//             for (int k = 0; k < AUX; k++) {
//               sum += teov[CoOrdinate({i, e_mu, k})] *
//                      teov[CoOrdinate({j, f_mu, k})];
//             }
//             if (sum != 0.0) {
//               output_nnzs.insert(CoOrdinate({i, j, e_mu, f_mu}));
//             }
//           }
//         }
//       }
//     }
//     std::cout << "Number of nonzeros in output: " << output_nnzs.size() <<
//     std::endl; Tensor<double> result = teov.multiply<double>(
//         teov, CoOrdinate({2}), CoOrdinate({}), CoOrdinate({2}),
//         CoOrdinate({}));
//     std::cout << "Number of nonzeros in output: " <<
//     result.get_nonzeros().size() << std::endl; tsl::hopscotch_set<CoOrdinate>
//     result_nnzs; for(auto nnz : result.get_nonzeros()){
//         result_nnzs.insert(nnz.get_coords());
//     }
//     assert(output_nnzs == result_nnzs);
// }
//
//
void run_oom_cases() {
  Tensor<double> frostt_tensor = Tensor<double>(
      "/media/saurabh/New Volume/ubuntu_downloads/frostt/flicrk-4d.tns", true);
  frostt_tensor._infer_dimensionality();
  frostt_tensor._infer_shape();
  // int l2_size = 16 * 1024 * 1024; //in bytes
  // int num_elts = l2_size / sizeof(double);
  // int tile_size = sqrt(scaling_factor * num_elts);
  int minsize = 1000000;
  int tile_size = sqrt(scaling_factor * minsize);
  // nips experiments
  std::cout << "Running flickr-4d tensor" << std::endl;
  if (fork() == 0) {
    std::cout << "mode 023 contraction" << std::endl;
    std::chrono::high_resolution_clock::time_point t1 =
        std::chrono::high_resolution_clock::now();
    ListTensor<double> result =
        frostt_tensor.parallel_tile2d_outer_multiply<double>(
            frostt_tensor, CoOrdinate({0, 2, 3}), CoOrdinate({0, 2, 3}),
            tile_size);
    std::chrono::high_resolution_clock::time_point t2 =
        std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "Flickr-4d mode 023: " << time_span.count() << " seconds."
              << std::endl;
    exit(0);
  } else {
    int stat;
    wait(&stat);
  }
  if (fork() == 0) {
    std::cout << "mode 013 contraction" << std::endl;
    std::chrono::high_resolution_clock::time_point t1 =
        std::chrono::high_resolution_clock::now();
    ListTensor<double> result =
        frostt_tensor.parallel_tile2d_outer_multiply<double>(
            frostt_tensor, CoOrdinate({0, 1, 3}), CoOrdinate({0, 1, 3}),
            tile_size);
    std::chrono::high_resolution_clock::time_point t2 =
        std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "Flickr-4d mode 013: " << time_span.count() << " seconds."
              << std::endl;
    exit(0);
  } else {
    int stat;
    wait(&stat);
  }
  if (fork() == 0) {
    std::cout << "mode 123 contraction" << std::endl;
    std::chrono::high_resolution_clock::time_point t1 =
        std::chrono::high_resolution_clock::now();
    ListTensor<double> result =
        frostt_tensor.parallel_tile2d_outer_multiply<double>(
            frostt_tensor, CoOrdinate({1, 2, 3}), CoOrdinate({1, 2, 3}),
            tile_size);
    std::chrono::high_resolution_clock::time_point t2 =
        std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "Flickr-4d mode 123: " << time_span.count() << " seconds."
              << std::endl;
    exit(0);
  } else {
    int stat;
    wait(&stat);
  }

  if (fork() == 0) {
    std::cout << "mode 02 contraction" << std::endl;
    std::chrono::high_resolution_clock::time_point t1 =
        std::chrono::high_resolution_clock::now();
    ListTensor<double> result =
        frostt_tensor.parallel_tile2d_outer_multiply<double>(
            frostt_tensor, CoOrdinate({0, 2}), CoOrdinate({0, 2}), tile_size);
    std::chrono::high_resolution_clock::time_point t2 =
        std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "Flickr-4d mode 02: " << time_span.count() << " seconds."
              << std::endl;
    exit(0);
  } else {
    int stat;
    wait(&stat);
  }
  if (fork() == 0) {
    std::cout << "mode 12 contraction" << std::endl;
    std::chrono::high_resolution_clock::time_point t1 =
        std::chrono::high_resolution_clock::now();
    ListTensor<double> result =
        frostt_tensor.parallel_tile2d_outer_multiply<double>(
            frostt_tensor, CoOrdinate({1, 2}), CoOrdinate({1, 2}), tile_size);
    std::chrono::high_resolution_clock::time_point t2 =
        std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "Flickr-4d mode 12: " << time_span.count() << " seconds."
              << std::endl;
    exit(0);
  } else {
    int stat;
    wait(&stat);
  }
  std::cout << "Running nell-2 tensor" << std::endl;
  frostt_tensor = Tensor<double>(
      "/media/saurabh/New Volume/ubuntu_downloads/frostt/nell-2.tns", true);
  frostt_tensor._infer_dimensionality();
  frostt_tensor._infer_shape();
  if (fork() == 0) {
    std::cout << "mode 02 contraction" << std::endl;
    std::chrono::high_resolution_clock::time_point t1 =
        std::chrono::high_resolution_clock::now();
    ListTensor<double> result =
        frostt_tensor.parallel_tile2d_outer_multiply<double>(
            frostt_tensor, CoOrdinate({0, 2}), CoOrdinate({0, 2}), tile_size);
    std::chrono::high_resolution_clock::time_point t2 =
        std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "Nell-2 mode 02: " << time_span.count() << " seconds."
              << std::endl;
    exit(0);
  } else {
    int stat;
    wait(&stat);
  }
  if (fork() == 0) {
    std::cout << "mode 12 contraction" << std::endl;
    std::chrono::high_resolution_clock::time_point t1 =
        std::chrono::high_resolution_clock::now();
    ListTensor<double> result =
        frostt_tensor.parallel_tile2d_outer_multiply<double>(
            frostt_tensor, CoOrdinate({1, 2}), CoOrdinate({1, 2}), tile_size);
    std::chrono::high_resolution_clock::time_point t2 =
        std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "Nell-2 mode 12: " << time_span.count() << " seconds."
              << std::endl;
    exit(0);
  } else {
    int stat;
    wait(&stat);
  }
  if (fork() == 0) {
    std::cout << "mode 01 contraction" << std::endl;
    std::chrono::high_resolution_clock::time_point t1 =
        std::chrono::high_resolution_clock::now();
    ListTensor<double> result =
        frostt_tensor.parallel_tile2d_outer_multiply<double>(
            frostt_tensor, CoOrdinate({0, 1}), CoOrdinate({0, 1}), tile_size);
    std::chrono::high_resolution_clock::time_point t2 =
        std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "Nell-2 mode 01: " << time_span.count() << " seconds."
              << std::endl;
    exit(0);
  } else {
    int stat;
    wait(&stat);
  }
}

void frostt_nnz_estimate() {
  std::string prefix = "/media/saurabh/New Volume1/ubuntu_downloads/frostt/";
  Tensor<double> frostt_tensor =
      Tensor<double>(prefix + "/chicago-crime-comm.tns", true);
  frostt_tensor._infer_dimensionality();
  frostt_tensor._infer_shape();
  std::cout << "Chicago crime comm tensor output nnz:" << std::endl;
  std::cout << "Mode 0 contraction: " << std::endl;
  std::cout << "using inner outer" << std::endl;
  auto bounds = frostt_tensor.bound_output_nnz_inner_outer(frostt_tensor, CoOrdinate({0}),
                                               CoOrdinate({0}));
  std::cout << "Lower bound: " << bounds.first
            << " Upper bound: " << bounds.second << std::endl;
  std::cout << "using outer outer" << std::endl;
  bounds = frostt_tensor.bound_output_nnz_outer_outer(frostt_tensor, CoOrdinate({0}),
                                               CoOrdinate({0}));
  std::cout << "Lower bound: " << bounds.first
            << " Upper bound: " << bounds.second << std::endl;
  std::cout << "Mode 0, 1 contraction: " << std::endl;
  std::cout << "using inner outer" << std::endl;
  bounds = frostt_tensor.bound_output_nnz_inner_outer(frostt_tensor, CoOrdinate({0, 1}),
                                          CoOrdinate({0, 1}));
  std::cout << "Lower bound: " << bounds.first
            << " Upper bound: " << bounds.second << std::endl;
  std::cout << "using outer outer" << std::endl;
  bounds = frostt_tensor.bound_output_nnz_outer_outer(frostt_tensor, CoOrdinate({0, 1}),
                                               CoOrdinate({0, 1}));
  std::cout << "Lower bound: " << bounds.first
            << " Upper bound: " << bounds.second << std::endl;
  std::cout << "Mode 1, 2, 3 contraction: " << std::endl;
  std::cout << "using inner outer" << std::endl;
  bounds = frostt_tensor.bound_output_nnz_inner_outer(frostt_tensor, CoOrdinate({1, 2, 3}),
                                          CoOrdinate({1, 2, 3}));
  std::cout << "Lower bound: " << bounds.first
            << " Upper bound: " << bounds.second << std::endl;
  std::cout << "using outer outer" << std::endl;
  bounds = frostt_tensor.bound_output_nnz_outer_outer(frostt_tensor, CoOrdinate({1, 2, 3}),
                                               CoOrdinate({1, 2, 3}));
  std::cout << "Lower bound: " << bounds.first
            << " Upper bound: " << bounds.second << std::endl;

  frostt_tensor = Tensor<double>(prefix + "/uber.tns", true);
  frostt_tensor._infer_dimensionality();
  frostt_tensor._infer_shape();
  std::cout << "Uber tensor output nnz:" << std::endl;
  std::cout << "Mode 0 2 contraction: " << std::endl;
  std::cout << "using inner outer" << std::endl;
  bounds = frostt_tensor.bound_output_nnz_inner_outer(frostt_tensor, CoOrdinate({0, 2}),
                                          CoOrdinate({0, 2}));
  std::cout << "Lower bound: " << bounds.first
            << " Upper bound: " << bounds.second << std::endl;
  std::cout << "using outer outer" << std::endl;
  bounds = frostt_tensor.bound_output_nnz_outer_outer(frostt_tensor, CoOrdinate({0, 2}),
                                               CoOrdinate({0, 2}));
  std::cout << "Lower bound: " << bounds.first
            << " Upper bound: " << bounds.second << std::endl;
  std::cout << "Mode 1 2 3 contraction: " << std::endl;
  std::cout << "using inner outer" << std::endl;
  bounds = frostt_tensor.bound_output_nnz_inner_outer(frostt_tensor, CoOrdinate({1, 2, 3}),
                                          CoOrdinate({1, 2, 3}));
  std::cout << "Lower bound: " << bounds.first
            << " Upper bound: " << bounds.second << std::endl;
  std::cout << "using outer outer" << std::endl;
  bounds = frostt_tensor.bound_output_nnz_outer_outer(frostt_tensor, CoOrdinate({1, 2, 3}),
                                               CoOrdinate({1, 2, 3}));
  std::cout << "Lower bound: " << bounds.first
            << " Upper bound: " << bounds.second << std::endl;
  frostt_tensor = Tensor<double>(prefix + "/vast-2015-mc1-5d.tns", true);
  frostt_tensor._infer_dimensionality();
  frostt_tensor._infer_shape();
  std::cout << "vast-5d tensor output nnz:" << std::endl;
  std::cout << "Mode 0 1 contraction: " << std::endl;
  std::cout << "using inner outer" << std::endl;
  bounds = frostt_tensor.bound_output_nnz_inner_outer(frostt_tensor, CoOrdinate({0, 1}),
                                          CoOrdinate({0, 1}));
  std::cout << "Lower bound: " << bounds.first
            << " Upper bound: " << bounds.second << std::endl;
  std::cout << "using outer outer" << std::endl;
  bounds = frostt_tensor.bound_output_nnz_outer_outer(frostt_tensor, CoOrdinate({0, 1}),
                                               CoOrdinate({0, 1}));
  std::cout << "Lower bound: " << bounds.first
            << " Upper bound: " << bounds.second << std::endl;
  std::cout << "Mode 0 1 4 contraction: " << std::endl;
  std::cout << "using inner outer" << std::endl;
  bounds = frostt_tensor.bound_output_nnz_inner_outer(frostt_tensor, CoOrdinate({0, 1, 4}),
                                          CoOrdinate({0, 1, 4}));
  std::cout << "Lower bound: " << bounds.first
            << " Upper bound: " << bounds.second << std::endl;
  std::cout << "using outer outer" << std::endl;
  bounds = frostt_tensor.bound_output_nnz_outer_outer(frostt_tensor, CoOrdinate({0, 1, 4}),
                                               CoOrdinate({0, 1, 4}));
  std::cout << "Lower bound: " << bounds.first
            << " Upper bound: " << bounds.second << std::endl;
  frostt_tensor = Tensor<double>(prefix + "/nips.tns", true);
  frostt_tensor._infer_dimensionality();
  frostt_tensor._infer_shape();
  std::cout << "nips tensor output nnz:" << std::endl;
  std::cout << "Mode 2 contraction: " << std::endl;
  std::cout << "using inner outer" << std::endl;
  bounds = frostt_tensor.bound_output_nnz_inner_outer(frostt_tensor, CoOrdinate({2}),
                                          CoOrdinate({2}));
  std::cout << "Lower bound: " << bounds.first
            << " Upper bound: " << bounds.second << std::endl;
  std::cout << "using outer outer" << std::endl;
  bounds = frostt_tensor.bound_output_nnz_outer_outer(frostt_tensor, CoOrdinate({2}),
                                               CoOrdinate({2}));
  std::cout << "Lower bound: " << bounds.first
            << " Upper bound: " << bounds.second << std::endl;
  std::cout << "Mode 2 3 contraction: " << std::endl;
  std::cout << "using inner outer" << std::endl;
  bounds = frostt_tensor.bound_output_nnz_inner_outer(frostt_tensor, CoOrdinate({2, 3}),
                                          CoOrdinate({2, 3}));
  std::cout << "Lower bound: " << bounds.first
            << " Upper bound: " << bounds.second << std::endl;
  std::cout << "using outer outer" << std::endl;
  bounds = frostt_tensor.bound_output_nnz_outer_outer(frostt_tensor, CoOrdinate({2, 3}),
                                               CoOrdinate({2, 3}));
  std::cout << "Lower bound: " << bounds.first
            << " Upper bound: " << bounds.second << std::endl;
  std::cout << "Mode 0 1 3 contraction: " << std::endl;
  std::cout << "using inner outer" << std::endl;
  bounds = frostt_tensor.bound_output_nnz_inner_outer(frostt_tensor, CoOrdinate({0, 1, 3}),
                                          CoOrdinate({0, 1, 3}));
  std::cout << "Lower bound: " << bounds.first
            << " Upper bound: " << bounds.second << std::endl;
  std::cout << "using outer outer" << std::endl;
  bounds = frostt_tensor.bound_output_nnz_outer_outer(frostt_tensor, CoOrdinate({0, 1, 3}),
                                               CoOrdinate({0, 1, 3}));
  std::cout << "Lower bound: " << bounds.first
            << " Upper bound: " << bounds.second << std::endl;
  return;
}

void run_frostt_experiments() {
  struct rusage usage_before, usage_after;
  std::string prefix = "/media/saurabh/New Volume1/ubuntu_downloads/frostt/";
  Tensor<double> frostt_tensor = Tensor<double>(prefix+"nips.tns", true);
  frostt_tensor._infer_dimensionality();
  frostt_tensor._infer_shape();
  //int l2_size = 16 * 1024 * 1024; //in bytes
  //int num_elts = l2_size / sizeof(double);
  //int tile_size = sqrt(scaling_factor * num_elts);
  int minsize = 1000000;
  int tile_size = sqrt(scaling_factor * minsize);
  // nips experiments
  std::cout << "Running nips tensor" << std::endl;
  int pid = fork();
  if (pid == 0) {
    std::cout << "mode 2 contraction" << std::endl;
    getrusage(RUSAGE_SELF, &usage_before);
    std::chrono::high_resolution_clock::time_point t1 =
        std::chrono::high_resolution_clock::now();
    ListTensor<double> result =
    frostt_tensor.parallel_HT_tile2d_outer_multiply<double>(
        frostt_tensor, CoOrdinate({2}), CoOrdinate({2}), tile_size);
    std::chrono::high_resolution_clock::time_point t2 =
        std::chrono::high_resolution_clock::now();
    getrusage(RUSAGE_SELF, &usage_after);
    std::chrono::duration<double> time_span =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "RAM usage (in KB): "
              << usage_after.ru_maxrss - usage_before.ru_maxrss << std::endl;
    std::cout << "NIPS mode 2: " << time_span.count() << " seconds."
              << std::endl;
    exit(0);
  } else {
    int stat;
    wait(&stat);
  }
  if (fork() == 0) {
    std::cout << "mode 2 3 contraction" << std::endl;
    getrusage(RUSAGE_SELF, &usage_before);
    std::chrono::high_resolution_clock::time_point t1 =
        std::chrono::high_resolution_clock::now();
    ListTensor<double> result = frostt_tensor.parallel_HT_tile2d_outer_multiply<double>(
        frostt_tensor, CoOrdinate({2, 3}), CoOrdinate({2, 3}), tile_size);
    std::chrono::high_resolution_clock::time_point t2 =
        std::chrono::high_resolution_clock::now();
    getrusage(RUSAGE_SELF, &usage_after);
    std::chrono::duration<double> time_span =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "NIPS mode 2, 3: " << time_span.count() << " seconds."
              << std::endl;
    std::cout << "RAM usage (in KB): "
              << usage_after.ru_maxrss - usage_before.ru_maxrss << std::endl;
    exit(0);
  } else {
    int stat;
    wait(&stat);
  }
  if (fork() == 0) {
    std::cout << "mode 0 1 3 contraction" << std::endl;
    getrusage(RUSAGE_SELF, &usage_before);
    std::chrono::high_resolution_clock::time_point t1 =
        std::chrono::high_resolution_clock::now();
    ListTensor<double> result = frostt_tensor.parallel_HT_tile2d_outer_multiply<double>(
        frostt_tensor, CoOrdinate({0, 1, 3}), CoOrdinate({0, 1, 3}), tile_size);
    std::chrono::high_resolution_clock::time_point t2 =
        std::chrono::high_resolution_clock::now();
    getrusage(RUSAGE_SELF, &usage_after);
    std::cout << "RAM usage (in KB): "
              << usage_after.ru_maxrss - usage_before.ru_maxrss << std::endl;
    std::chrono::duration<double> time_span =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "NIPS mode 0, 1, 3: " << time_span.count() << " seconds."
              << std::endl;
    exit(0);
  } else {
    int stat;
    wait(&stat);
  }
  //std::string prefix = "/media/saurabh/New Volume/ubuntu_downloads/frostt/";

  ////////// chicago experiments
  std::cout << "Running chicago tensor" << std::endl;
  frostt_tensor =
      Tensor<double>(prefix+"/chicago-crime-comm.tns",
                     true);
  frostt_tensor._infer_dimensionality();
  frostt_tensor._infer_shape();
  if (fork() == 0) {
    std::cout << "mode 0 contraction" << std::endl;
    getrusage(RUSAGE_SELF, &usage_before);
    std::chrono::high_resolution_clock::time_point t1 =
        std::chrono::high_resolution_clock::now();
    ListTensor<double> result = frostt_tensor.parallel_HT_tile2d_outer_multiply<double>(
        frostt_tensor, CoOrdinate({0}), CoOrdinate({0}), tile_size);
    std::chrono::high_resolution_clock::time_point t2 =
        std::chrono::high_resolution_clock::now();
    getrusage(RUSAGE_SELF, &usage_after);
    std::cout << "RAM usage (in KB): "
              << usage_after.ru_maxrss - usage_before.ru_maxrss << std::endl;
    std::chrono::duration<double> time_span =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "chicago mode 0: " << time_span.count() << " seconds."
              << std::endl;
    exit(0);
  } else {
    int stat;
    wait(&stat);
  }
  if (fork() == 0) {
    std::cout << "mode 0 1 contraction" << std::endl;
    getrusage(RUSAGE_SELF, &usage_before);
    std::chrono::high_resolution_clock::time_point t1 =
        std::chrono::high_resolution_clock::now();
    ListTensor<double> result = frostt_tensor.parallel_HT_tile2d_outer_multiply<double>(
        frostt_tensor, CoOrdinate({0, 1}), CoOrdinate({0, 1}), tile_size);
    std::chrono::high_resolution_clock::time_point t2 =
        std::chrono::high_resolution_clock::now();
    getrusage(RUSAGE_SELF, &usage_after);
    std::cout << "RAM usage (in KB): "
              << usage_after.ru_maxrss - usage_before.ru_maxrss << std::endl;
    std::chrono::duration<double> time_span =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "chicago mode 0, 1: " << time_span.count() << " seconds."
              << std::endl;
    exit(0);
  } else {
    int stat;
    wait(&stat);
  }
  if (fork() == 0) {
    std::cout << "mode 1 2 3 contraction" << std::endl;
    getrusage(RUSAGE_SELF, &usage_before);
    std::chrono::high_resolution_clock::time_point t1 =
        std::chrono::high_resolution_clock::now();
    ListTensor<double> result = frostt_tensor.parallel_HT_tile2d_outer_multiply<double>(
        frostt_tensor, CoOrdinate({1, 2, 3}), CoOrdinate({1, 2, 3}), tile_size);
    std::chrono::high_resolution_clock::time_point t2 =
        std::chrono::high_resolution_clock::now();
    getrusage(RUSAGE_SELF, &usage_after);
    std::cout << "RAM usage (in KB): "
              << usage_after.ru_maxrss - usage_before.ru_maxrss << std::endl;
    std::chrono::duration<double> time_span =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "chicago mode 1, 2, 3: " << time_span.count() << " seconds."
              << std::endl;
    exit(0);
  } else {
    int stat;
    wait(&stat);
  }

  //////////// vast-3d experiments
  std::cout << "Running vast-5d tensor" << std::endl;
  frostt_tensor =
      Tensor<double>(prefix+"/vast-2015-mc1-5d.tns",
                     true);
  frostt_tensor._infer_dimensionality();
  frostt_tensor._infer_shape();
  if (fork() == 0) {
    std::cout << "mode 0 1 contraction" << std::endl;
    getrusage(RUSAGE_SELF, &usage_before);
    std::chrono::high_resolution_clock::time_point t1 =
        std::chrono::high_resolution_clock::now();
    ListTensor<double> result = frostt_tensor.parallel_HT_tile2d_outer_multiply<double>(
        frostt_tensor, CoOrdinate({0, 1}), CoOrdinate({0, 1}), tile_size);
    std::chrono::high_resolution_clock::time_point t2 =
        std::chrono::high_resolution_clock::now();
    getrusage(RUSAGE_SELF, &usage_after);
    std::cout << "RAM usage (in GB): "
              << (usage_after.ru_maxrss - usage_before.ru_maxrss) << std::endl;
    std::chrono::duration<double> time_span =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "vast-2015-mc1-5d mode 0, 1: " << time_span.count() << " seconds."
              << std::endl;
    exit(0);
  } else {
    int stat;
    wait(&stat);
  }
  if (fork() == 0) {
    std::cout << "mode 0 1 4 contraction" << std::endl;
    getrusage(RUSAGE_SELF, &usage_before);
    std::chrono::high_resolution_clock::time_point t1 =
        std::chrono::high_resolution_clock::now();
    ListTensor<double> result =
        frostt_tensor.parallel_HT_tile2d_outer_multiply<double>(
            frostt_tensor, CoOrdinate({0, 1, 4}), CoOrdinate({0, 1, 4}), tile_size);
    std::chrono::high_resolution_clock::time_point t2 =
        std::chrono::high_resolution_clock::now();
    getrusage(RUSAGE_SELF, &usage_after);
    std::cout << "RAM usage (in GB): "
              << (usage_after.ru_maxrss - usage_before.ru_maxrss) << std::endl;
    std::chrono::duration<double> time_span =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "vast-2015-mc1-5d mode 0, 1, 4: " << time_span.count()
              << " seconds." << std::endl;
    exit(0);
  } else {
    int stat;
    wait(&stat);
  }
  ////
  ///////////// uber experiments
  std::cout << "Running uber tensor" << std::endl;
  frostt_tensor =
      Tensor<double>(prefix+"/uber.tns",
                     true);
  frostt_tensor._infer_dimensionality();
  frostt_tensor._infer_shape();
  if (fork() == 0) {
    std::cout << "mode 0 2 contraction" << std::endl;
    getrusage(RUSAGE_SELF, &usage_before);
    std::chrono::high_resolution_clock::time_point t1 =
        std::chrono::high_resolution_clock::now();
    ListTensor<double> result = frostt_tensor.parallel_HT_tile2d_outer_multiply<double>(
        frostt_tensor, CoOrdinate({0, 2}), CoOrdinate({0, 2}), tile_size);
    std::chrono::high_resolution_clock::time_point t2 =
        std::chrono::high_resolution_clock::now();
    getrusage(RUSAGE_SELF, &usage_after);
    std::cout << "RAM usage (in GB): "
              << (usage_after.ru_maxrss - usage_before.ru_maxrss) << std::endl;
    std::chrono::duration<double> time_span =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "uber mode 0 2: " << time_span.count() << " seconds."
              << std::endl;
    exit(0);
  } else {
    int stat;
    wait(&stat);
  }
  if (fork() == 0) {
    std::cout << "mode 1 2 3 contraction" << std::endl;
    getrusage(RUSAGE_SELF, &usage_before);
    std::chrono::high_resolution_clock::time_point t1 =
        std::chrono::high_resolution_clock::now();
    ListTensor<double> result = frostt_tensor.parallel_HT_tile2d_outer_multiply<double>(
        frostt_tensor, CoOrdinate({1, 2, 3}), CoOrdinate({1, 2, 3}), tile_size);
    std::chrono::high_resolution_clock::time_point t2 =
        std::chrono::high_resolution_clock::now();
    getrusage(RUSAGE_SELF, &usage_after);
    std::cout << "RAM usage (in GB): "
              << (usage_after.ru_maxrss - usage_before.ru_maxrss) << std::endl;
    std::chrono::duration<double> time_span =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "uber mode 1, 2, 3: " << time_span.count() << " seconds."
              << std::endl;
    exit(0);
  } else {
    int stat;
    wait(&stat);
  }
}

int get_l2_num_elts(){
    int l2_bytes = 2 * 1024 * 1024;
    int l2_numelts = l2_bytes / sizeof(double);
    return l2_numelts;
}

void run_dlpno_experiments() {
    std::cout << "Running helium data small" << std::endl;
    int minsize = 100000;
    int tile_size = sqrt(scaling_factor * minsize);
    //if (fork() == 0) {
    //    Tensor<double> teov = Tensor<double>(
    //        "./helium_small/threec_int.tns",
    //        true);
    //    teov._infer_dimensionality();
    //    teov._infer_shape();
    //    std::cout << "Time for TEov * TEov " << std::endl;
    //    std::chrono::high_resolution_clock::time_point t1 =
    //        std::chrono::high_resolution_clock::now();
    //    ListTensor<double> result = teov.parallel_tile2d_outer_multiply<double>(teov, CoOrdinate({0}),
    //                                                CoOrdinate({0}), tile_size);
    //    std::chrono::high_resolution_clock::time_point t2 =
    //        std::chrono::high_resolution_clock::now();
    //    std::chrono::duration<double> time_span =
    //        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    //    std::cout << "helium small time : " << time_span.count() << " seconds."
    //              << std::endl;
    //    exit(0);
    //} else {
    //    int stat;
    //    wait(&stat);
    //}
    //std::cout << "Running helium data medium" << std::endl;
    //if (fork() == 0) {
    //    Tensor<double> teov = Tensor<double>(
    //        "./helium_medium/threec_int.tns",
    //        true);
    //    teov._infer_dimensionality();
    //    teov._infer_shape();
    //    std::cout << "Time for TEov * TEov " << std::endl;
    //    std::chrono::high_resolution_clock::time_point t1 =
    //        std::chrono::high_resolution_clock::now();
    //    ListTensor<double> result = teov.parallel_tile2d_outer_multiply<double>(teov, CoOrdinate({0}),
    //                                                CoOrdinate({0}), tile_size);
    //    std::chrono::high_resolution_clock::time_point t2 =
    //        std::chrono::high_resolution_clock::now();
    //    std::chrono::duration<double> time_span =
    //        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    //    std::cout << "helium medium time : " << time_span.count() << " seconds."
    //              << std::endl;
    //    exit(0);
    //} else {
    //    int stat;
    //    wait(&stat);
    //}
    std::cout << "Running helium data large" << std::endl;
    if (fork() == 0) {
        Tensor<double> teov = Tensor<double>(
            "./helium_large/threec_int.tns",
            true);
        teov._infer_dimensionality();
        teov._infer_shape();
        std::cout << "Time for TEov * TEov " << std::endl;
        std::chrono::high_resolution_clock::time_point t1 =
            std::chrono::high_resolution_clock::now();
        ListTensor<double> result = teov.parallel_tile2d_outer_multiply<double>(teov, CoOrdinate({0}),
                                                    CoOrdinate({0}), tile_size);
        std::chrono::high_resolution_clock::time_point t2 =
            std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span =
            std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
        std::cout << "helium large time : " << time_span.count() << " seconds."
                  << std::endl;
        exit(0);
    } else {
        int stat;
        wait(&stat);
    }
}

int main() {

  std::vector<float> tile_scaling_factors = {1,    5,     25, 125, 625, 78125, 390625};
  //std::vector<float> tile_scaling_factors = {0.25, 1, 4, 16, 64, 256, 1024, 4096};

  //std::vector<float> tile_scaling_factors = {125, 625, 78125, 390625};
  //std::vector<float> tile_scaling_factors = {3125,
  //                                           //15625,
  //                                           //78125,
  //                                           //390625,
  //                                           //1953125,
  //                                           //9765625,
  //                                           static_cast<float>(48828125),
  //                                           static_cast<float>(244140625)};
  //std::vector<float> tile_scaling_factors = {125};
  for (auto s : tile_scaling_factors) {
    scaling_factor = s;
    std::cout << "Scaling factor: " << scaling_factor << std::endl;
    //run_dlpno_experiments();
    run_frostt_experiments();
  }
  //run_frostt_experiments();
  //run_dlpno_experiments();
  //frostt_nnz_estimate();
}
