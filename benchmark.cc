#include "sparse_opcount/contract.hpp"
#include "sparse_opcount/read.hpp"
#include <chrono>
#include <iostream>
#include <sys/resource.h>
#include <sys/wait.h>
#include <vector>
static float scaling_factor = 0.5;


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
void run_frostt_experiments() {
  struct rusage usage_before, usage_after;
  //Tensor<double> frostt_tensor = Tensor<double>(
  //    "/media/saurabh/New Volume/ubuntu_downloads/frostt/nips.tns", true);
  //frostt_tensor._infer_dimensionality();
  //frostt_tensor._infer_shape();
  //int l2_size = 16 * 1024 * 1024; //in bytes
  //int num_elts = l2_size / sizeof(double);
  //int tile_size = sqrt(scaling_factor * num_elts);
  int minsize = 10000;
  int tile_size = sqrt(scaling_factor * minsize);
  // nips experiments
  //std::cout << "Running nips tensor" << std::endl;
  //int pid = fork();
  //if (pid == 0) {
  //  std::cout << "mode 2 contraction" << std::endl;
  //  getrusage(RUSAGE_SELF, &usage_before);
  //  std::chrono::high_resolution_clock::time_point t1 =
  //      std::chrono::high_resolution_clock::now();
  //  //frostt_tensor.microbench_tile2d(
  //  //    frostt_tensor, CoOrdinate({2}), CoOrdinate({2}), tile_size);
  //  //frostt_tensor.microbench_outer_outer(
  //  //    frostt_tensor, CoOrdinate({2}), CoOrdinate({2}));
  //  ListTensor<double> result =
  //  frostt_tensor.parallel_tile2d_outer_multiply<double>(
  //      frostt_tensor, CoOrdinate({2}), CoOrdinate({2}), tile_size);
  //  //  CompactTensor<double> result =
  //  //  frostt_tensor.parallel_inner_outer_multiply<double>(
  //  //     frostt_tensor, CoOrdinate({2}), CoOrdinate({}), CoOrdinate({2}),
  //  //     CoOrdinate({}));
  //  std::chrono::high_resolution_clock::time_point t2 =
  //      std::chrono::high_resolution_clock::now();
  //  getrusage(RUSAGE_SELF, &usage_after);
  //  std::chrono::duration<double> time_span =
  //      std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
  //  std::cout << "RAM usage (in KB): "
  //            << usage_after.ru_maxrss - usage_before.ru_maxrss << std::endl;
  //  std::cout << "NIPS mode 2: " << time_span.count() << " seconds."
  //            << std::endl;
  //  exit(0);
  //} else {
  //  int stat;
  //  wait(&stat);
  //}
  //if (fork() == 0) {
  //  std::cout << "mode 0 1 contraction" << std::endl;
  //  getrusage(RUSAGE_SELF, &usage_before);
  //  std::chrono::high_resolution_clock::time_point t1 =
  //      std::chrono::high_resolution_clock::now();
  //  ListTensor<double> result = frostt_tensor.parallel_tile2d_outer_multiply<double>(
  //      frostt_tensor, CoOrdinate({0, 1}), CoOrdinate({0, 1}), tile_size);
  //  //frostt_tensor.microbench_tile2d(
  //  //    frostt_tensor, CoOrdinate({0, 1}), CoOrdinate({0, 1}), tile_size);
  //  //frostt_tensor.microbench_outer_outer(
  //  //    frostt_tensor, CoOrdinate({0, 1}), CoOrdinate({0, 1}));
  //  //CompactTensor<double> result = frostt_tensor.inner_outer_multiply<double>(
  //  //    frostt_tensor, CoOrdinate({0, 1}), CoOrdinate({0, 1}));
  //  std::chrono::high_resolution_clock::time_point t2 =
  //      std::chrono::high_resolution_clock::now();
  //  getrusage(RUSAGE_SELF, &usage_after);
  //  std::chrono::duration<double> time_span =
  //      std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
  //  std::cout << "Tile size: 2" << std::endl;
  //  std::cout << "NIPS mode 0, 1: " << time_span.count() << " seconds."
  //            << std::endl;
  //  std::cout << "RAM usage (in KB): "
  //            << usage_after.ru_maxrss - usage_before.ru_maxrss << std::endl;
  //  exit(0);
  //} else {
  //  int stat;
  //  wait(&stat);
  //}
  //if (fork() == 0) {
  //  std::cout << "mode 1 2 contraction" << std::endl;
  //  getrusage(RUSAGE_SELF, &usage_before);
  //  std::chrono::high_resolution_clock::time_point t1 =
  //      std::chrono::high_resolution_clock::now();
  //  ListTensor<double> result = frostt_tensor.parallel_tile2d_outer_multiply<double>(
  //      frostt_tensor, CoOrdinate({1, 2}), CoOrdinate({1, 2}), tile_size);
  //  //frostt_tensor.microbench_tile2d(
  //  //    frostt_tensor, CoOrdinate({1, 2}), CoOrdinate({1, 2}), tile_size);
  //  //frostt_tensor.microbench_outer_outer(
  //  //    frostt_tensor, CoOrdinate({1, 2}), CoOrdinate({1, 2}));
  //  // CompactTensor<double> result =
  //  //     frostt_tensor.parallel_inner_outer_multiply<double>(
  //  //         frostt_tensor, CoOrdinate({1, 2}), CoOrdinate({}),
  //  //         CoOrdinate({1, 2}), CoOrdinate({}));
  //  //CompactTensor<double> result =
  //  //frostt_tensor.inner_outer_multiply<double>(
  //  //    frostt_tensor, CoOrdinate({1, 2}), CoOrdinate({1, 2}));
  //  std::chrono::high_resolution_clock::time_point t2 =
  //      std::chrono::high_resolution_clock::now();
  //  getrusage(RUSAGE_SELF, &usage_after);
  //  std::cout << "RAM usage (in KB): "
  //            << usage_after.ru_maxrss - usage_before.ru_maxrss << std::endl;
  //  std::chrono::duration<double> time_span =
  //      std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
  //  std::cout << "NIPS mode 1, 2: " << time_span.count() << " seconds."
  //            << std::endl;
  //  exit(0);
  //} else {
  //  int stat;
  //  wait(&stat);
  //}
  std::string prefix = "/media/saurabh/New Volume/ubuntu_downloads/frostt/";

  ////// chicago experiments
  std::cout << "Running chicago tensor" << std::endl;
  Tensor<double> frostt_tensor =
      Tensor<double>(prefix+"/chicago-crime-comm.tns",
                     true);
  frostt_tensor._infer_dimensionality();
  frostt_tensor._infer_shape();
  if (fork() == 0) {
    std::cout << "mode 0 contraction" << std::endl;
    getrusage(RUSAGE_SELF, &usage_before);
    std::chrono::high_resolution_clock::time_point t1 =
        std::chrono::high_resolution_clock::now();
    //frostt_tensor.microbench_tile2d(
    //    frostt_tensor, CoOrdinate({0}), CoOrdinate({0}), tile_size);
    //frostt_tensor.microbench_outer_outer(
    //    frostt_tensor, CoOrdinate({0}), CoOrdinate({0}));
    //ListTensor<double> result = frostt_tensor.parallel_tile2d_outer_multiply<double>(
    //    frostt_tensor, CoOrdinate({0}), CoOrdinate({0}), tile_size);

    ListTensor<double> result = frostt_tensor.parallel_tile2d_outer_multiply<double>(
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
    ListTensor<double> result = frostt_tensor.parallel_tile2d_outer_multiply<double>(
        frostt_tensor, CoOrdinate({0, 1}), CoOrdinate({0, 1}));
    // CompactTensor<double> result =
    //    frostt_tensor.parallel_inner_outer_multiply<double>(
    //        frostt_tensor, CoOrdinate({0, 1}), CoOrdinate({}),
    //        CoOrdinate({0, 1}), CoOrdinate({}));
    //CompactTensor<double> result = frostt_tensor.inner_outer_multiply<double>(
    //    frostt_tensor, CoOrdinate({0, 1}), CoOrdinate({0, 1}));
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
    ListTensor<double> result = frostt_tensor.parallel_tile2d_outer_multiply<double>(
        frostt_tensor, CoOrdinate({1, 2, 3}), CoOrdinate({1, 2, 3}));
    // CompactTensor<double> result =
    //    frostt_tensor.parallel_inner_outer_multiply<double>(
    //        frostt_tensor, CoOrdinate({0, 1}), CoOrdinate({}),
    //        CoOrdinate({0, 1}), CoOrdinate({}));
    //CompactTensor<double> result = frostt_tensor.inner_outer_multiply<double>(
    //    frostt_tensor, CoOrdinate({0, 1}), CoOrdinate({0, 1}));
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

  ////////// vast-3d experiments
  std::cout << "Running vast-5d tensor" << std::endl;
  frostt_tensor =
      Tensor<double>(prefix+"/vast-2015-mc1-5d.tns",
                     true);
  frostt_tensor._infer_dimensionality();
  frostt_tensor._infer_shape();
  if (fork() == 0) {
    std::cout << "mode 1 3 contraction" << std::endl;
    getrusage(RUSAGE_SELF, &usage_before);
    std::chrono::high_resolution_clock::time_point t1 =
        std::chrono::high_resolution_clock::now();
    //frostt_tensor.microbench_tile2d(
    //    frostt_tensor, CoOrdinate({0}), CoOrdinate({0}), tile_size);
    //frostt_tensor.microbench_outer_outer(
    //    frostt_tensor, CoOrdinate({0}), CoOrdinate({0}));
    ListTensor<double> result = frostt_tensor.parallel_tile2d_outer_multiply<double>(
        frostt_tensor, CoOrdinate({1, 3}), CoOrdinate({1, 3}), tile_size);
    std::chrono::high_resolution_clock::time_point t2 =
        std::chrono::high_resolution_clock::now();
    getrusage(RUSAGE_SELF, &usage_after);
    std::cout << "RAM usage (in GB): "
              << (usage_after.ru_maxrss - usage_before.ru_maxrss) << std::endl;
    std::chrono::duration<double> time_span =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "vast-2015-mc1-3d mode 1, 2, 4: " << time_span.count() << " seconds."
              << std::endl;
    exit(0);
  } else {
    int stat;
    wait(&stat);
  }
  if (fork() == 0) {
    std::cout << "mode 1 2 4 contraction" << std::endl;
    getrusage(RUSAGE_SELF, &usage_before);
    std::chrono::high_resolution_clock::time_point t1 =
        std::chrono::high_resolution_clock::now();
    // frostt_tensor.microbench_tile2d(
    //     frostt_tensor, CoOrdinate({0}), CoOrdinate({0}), tile_size);
    // frostt_tensor.microbench_outer_outer(
    //     frostt_tensor, CoOrdinate({0}), CoOrdinate({0}));
    ListTensor<double> result =
        frostt_tensor.parallel_tile2d_outer_multiply<double>(
            frostt_tensor, CoOrdinate({1, 2, 4}), CoOrdinate({1, 2, 4}), tile_size);
    std::chrono::high_resolution_clock::time_point t2 =
        std::chrono::high_resolution_clock::now();
    getrusage(RUSAGE_SELF, &usage_after);
    std::cout << "RAM usage (in GB): "
              << (usage_after.ru_maxrss - usage_before.ru_maxrss) << std::endl;
    std::chrono::duration<double> time_span =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "vast-2015-mc1-5d mode 1, 2, 4: " << time_span.count()
              << " seconds." << std::endl;
    exit(0);
  } else {
    int stat;
    wait(&stat);
  }
  //
  /////////// uber experiments
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
    //frostt_tensor.microbench_tile2d(
    //    frostt_tensor, CoOrdinate({0, 3}), CoOrdinate({0, 3}), tile_size);
    //frostt_tensor.microbench_outer_outer(
    //    frostt_tensor, CoOrdinate({0}), CoOrdinate({0}));
    ListTensor<double> result = frostt_tensor.parallel_tile2d_outer_multiply<double>(
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
    //frostt_tensor.microbench_tile2d(
    //    frostt_tensor, CoOrdinate({0, 3}), CoOrdinate({0, 3}), tile_size);
    //frostt_tensor.microbench_outer_outer(
    //    frostt_tensor, CoOrdinate({0}), CoOrdinate({0}));
    ListTensor<double> result = frostt_tensor.parallel_tile2d_outer_multiply<double>(
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

double self_contraction(Tensor<double> dlpno_tensor) {
  dlpno_tensor._infer_dimensionality();
  dlpno_tensor._infer_shape();
  std::chrono::high_resolution_clock::time_point t1 =
      std::chrono::high_resolution_clock::now();
  std::chrono::high_resolution_clock::time_point t2 =
      std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time_span =
      std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
  //int l2_size = 16 * 1024 * 1024; //in bytes
  //int num_elts = l2_size / sizeof(double);
  //int tile_size = scaling_factor * num_elts;
  int minsize = 10000;
  int tile_size = sqrt(scaling_factor * minsize);
  int sqrt_tile_size = tile_size;
  t1 = std::chrono::high_resolution_clock::now();
  // CompactTensor<double> result_inout =
  //    dlpno_tensor.inner_outer_multiply<double>(dlpno_tensor, CoOrdinate({2}),
  //                                              CoOrdinate({2}));
  ListTensor<double> result_inout = dlpno_tensor.parallel_tile2d_outer_multiply<double>(
      dlpno_tensor, CoOrdinate({2}), CoOrdinate({2}), sqrt_tile_size);
  t2 = std::chrono::high_resolution_clock::now();
  time_span =
      std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
  std::cout << time_span.count() << " seconds." << std::endl;
  return time_span.count();
}

double pair_contraction(Tensor<double> dlpno_tensor1,
                        Tensor<double> dlpno_tensor2) {
  // res(i, j, m, e_mu) = teoo(i, j, k) * teov(m, e_mu, k)
  dlpno_tensor1._infer_dimensionality();
  dlpno_tensor1._infer_shape();
  dlpno_tensor2._infer_dimensionality();
  dlpno_tensor2._infer_shape();
  std::chrono::high_resolution_clock::time_point t1 =
      std::chrono::high_resolution_clock::now();
  std::chrono::high_resolution_clock::time_point t2 =
      std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time_span =
      std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
  //int l2_size = 16 * 1024 * 1024; //in bytes
  //int num_elts = l2_size / sizeof(double);
  //int tile_size = scaling_factor * num_elts;
  int minsize = 10000;
  int tile_size = sqrt(scaling_factor * minsize);
  int sqrt_tile_size = tile_size;

  t1 = std::chrono::high_resolution_clock::now();
  // CompactTensor<double> result_inout =
  //    dlpno_tensor1.inner_outer_multiply<double>(
  //        dlpno_tensor2, CoOrdinate({2}), CoOrdinate({2}));
  // CompactTensor<double> result_inout =
  //     dlpno_tensor1.outer_outer_multiply<double>(
  //         dlpno_tensor2, CoOrdinate({2}), CoOrdinate({2}));
  ListTensor<double> result_inout =
      dlpno_tensor1.parallel_tile2d_outer_multiply<double>(
          dlpno_tensor2, CoOrdinate({2}), CoOrdinate({2}), sqrt_tile_size);

  t2 = std::chrono::high_resolution_clock::now();
  time_span =
      std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
  std::cout << time_span.count() << " seconds." << std::endl;
  return time_span.count();
}

void run_dlpno_experiments() {
  std::cout << "Running caffeine data" << std::endl;
  Tensor<double> teoo = Tensor<double>("./caffeine_data/TEoo.tns", true);
  teoo._infer_dimensionality();
  teoo._infer_shape();
  Tensor<double> teov = Tensor<double>("./caffeine_data/TEov.tns", true);
  teov._infer_dimensionality();
  teov._infer_shape();
  Tensor<double> tevv = Tensor<double>("./caffeine_data/TEvv.tns", true);
  tevv._infer_dimensionality();
  tevv._infer_shape();
  std::cout << "Time for TEov * TEov " << std::endl;
  self_contraction(teov);
  std::cout << "Time for TEoo * TEoo " << std::endl;
  self_contraction(teoo);
  std::cout << "Time for TEov * TEoo " << std::endl;
  pair_contraction(teov, teoo);
  std::cout << "Time for TEvv * TEoo " << std::endl;
  pair_contraction(tevv, teoo);
}

int main() {
  // taco::Tensor<double> teov =
  //     taco::read("./benzene_data/TEov.tns",
  //                taco::Format({taco::Dense, taco::Sparse, taco::Sparse}));
  // std::cout << teov.getDimension(0) << ", " << teov.getDimension(1) << ", "
  //           << teov.getDimension(2) << std::endl;
  // teov.pack();
  // taco::Tensor<double> teov1 =
  //     taco::read("./benzene_data/TEov.tns",
  //                taco::Format({taco::Dense, taco::Sparse, taco::Sparse}));
  // teov1.setName("teov1");
  // teov1.pack();
  // taco_teov_pao_contraction(teov, teov1);
  //  std::cout<<"int is "<<sizeof(int)<<", short is "<<sizeof(short)<<", long
  //  is
  //  "<<sizeof(long)<<", long long is "<<sizeof(long long)<<std::endl;
  // Tensor<double> teoo =
  //    Tensor<double>("./caffeine_data/TEoo.tns", true); //
  //    Tensor<double>("./benzene_data/TEov.tns", false);
  // Tensor<double> teov =
  //    Tensor<double>("./caffeine_data/TEov.tns", true); //
  //    Tensor<double>("./benzene_data/TEov.tns", false);
  // teoo_teov_k_contraction(teoo, teov);
  // teoo_k_contraction(teoo);
  // hmap_teov_pao_contraction(teov);
  // Tensor<double> tevv = Tensor<double>("./caffeine_data/TEvv.tns", true);
  // std::cout<<"gonna run tevv times teov" <<std::endl;
  // teoo_teov_k_contraction(tevv, teov);
  // std::cout<<"gonna run tevv times teoo" <<std::endl;
  // teoo_teov_k_contraction(tevv, teoo);
  // try tevv times teov.
  // Tensor<double> frostt_tensor = Tensor<double>(
  //    "/media/saurabh/New Volume1/ubuntu_downloads/frostt/nips.tns",
  //    true); // Tensor<double>("./benzene_data/TEov.tns", false);
  // single_mode_contraction(frostt_tensor);
  // two_mode_contraction(frostt_tensor);
  //std::vector<float> tile_scaling_factors = {0.04, 0.2, 1,    5,     25,
  //                                           125,  625, 3125, 15625, 78125};
  std::vector<float> tile_scaling_factors = {25, 125,  625, 3125};
  // 5 is too small
  for (auto s : tile_scaling_factors) {
    scaling_factor = s;
    std::cout << "Scaling factor: " << scaling_factor << std::endl;
    //run_dlpno_experiments();
    run_frostt_experiments();
  }
  //run_frostt_experiments();
  //taco::Tensor<double> frostt_tensor =
  //    taco::read("/media/saurabh/New Volume1/ubuntu_downloads/frostt/nips.tns",
  //               taco::Format({taco::Dense, taco::Sparse, taco::Sparse,
  //                             taco::Sparse}));
  //taco::Tensor<double> frostt_tensor =
  //    taco::read("/media/saurabh/New Volume1/ubuntu_downloads/frostt/nips.tns",
  //               taco::Format({taco::Dense, taco::Sparse, taco::Sparse,
  //                             taco::Sparse}));
  //taco_frostt(frostt_tensor);
  //run_dlpno_experiments();
}
