#include "sparse_opcount/contract.hpp"
#include "sparse_opcount/read.hpp"
#include <chrono>
#include <iostream>
#include <sys/resource.h>
#include <sys/wait.h>
#include <vector>
static float scaling_factor = 1;
std::map<std::string, std::vector<float>> averages_per_tensor;


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

void make_frostt_signatures(std::vector<int>& tile_sizes) {
  //std::string prefix = "/media/saurabh/New Volume1/ubuntu_downloads/frostt/";
  std::string prefix = "/home/hunter/work/frostt";
  Tensor<double> frostt = Tensor<double>(prefix + "/nips.tns", true);
  frostt._infer_dimensionality();
  frostt._infer_shape();
  if (averages_per_tensor.find("nips013") == averages_per_tensor.end()) {
    averages_per_tensor["nips013"] = std::vector<float>();
  }
  for (auto tile_size : tile_sizes) {
    averages_per_tensor["nips013"].push_back(
        frostt.total_active_columns(CoOrdinate({0, 1, 3}), tile_size));
  }

  frostt = Tensor<double>(prefix + "/chicago-crime-comm.tns", true);
  frostt._infer_dimensionality();
  frostt._infer_shape();
  if (averages_per_tensor.find("chicago0") == averages_per_tensor.end()) {
    averages_per_tensor["chicago0"] = std::vector<float>();
  }
  for (auto tile_size : tile_sizes) {
    averages_per_tensor["chicago0"].push_back(
        frostt.total_active_columns(CoOrdinate({0}), tile_size));
  }

  frostt = Tensor<double>(prefix + "/chicago-crime-comm.tns", true);
  frostt._infer_dimensionality();
  frostt._infer_shape();
  if (averages_per_tensor.find("chicago01") == averages_per_tensor.end()) {
    averages_per_tensor["chicago01"] = std::vector<float>();
  }
  for (auto tile_size : tile_sizes) {
    averages_per_tensor["chicago01"].push_back(
        frostt.total_active_columns(CoOrdinate({0, 1}), tile_size));
  }

  frostt = Tensor<double>(prefix + "/chicago-crime-comm.tns", true);
  frostt._infer_dimensionality();
  frostt._infer_shape();
  if (averages_per_tensor.find("chicago123") == averages_per_tensor.end()) {
    averages_per_tensor["chicago123"] = std::vector<float>();
  }
  for (auto tile_size : tile_sizes) {
    averages_per_tensor["chicago123"].push_back(
        frostt.total_active_columns(CoOrdinate({1, 2, 3}), tile_size));
  }

  frostt = Tensor<double>(prefix + "/uber.tns", true);
  frostt._infer_dimensionality();
  frostt._infer_shape();
  if (averages_per_tensor.find("uber02") == averages_per_tensor.end()) {
    averages_per_tensor["uber02"] = std::vector<float>();
  }
  for (auto tile_size : tile_sizes) {
    averages_per_tensor["uber02"].push_back(
        frostt.total_active_columns(CoOrdinate({0, 2}), tile_size));
  }

  frostt = Tensor<double>(prefix + "/uber.tns", true);
  frostt._infer_dimensionality();
  frostt._infer_shape();
  if (averages_per_tensor.find("uber123") == averages_per_tensor.end()) {
    averages_per_tensor["uber123"] = std::vector<float>();
  }
  for (auto tile_size : tile_sizes) {
    averages_per_tensor["uber123"].push_back(
        frostt.total_active_columns(CoOrdinate({1, 2, 3}), tile_size));
  }

  frostt = Tensor<double>(prefix + "/vast-2015-mc1-5d.tns", true);
  frostt._infer_dimensionality();
  frostt._infer_shape();
  if (averages_per_tensor.find("vast01") == averages_per_tensor.end()) {
    averages_per_tensor["vast01"] = std::vector<float>();
  }
  for (auto tile_size : tile_sizes) {
    averages_per_tensor["vast01"].push_back(
        frostt.total_active_columns(CoOrdinate({0, 1}), tile_size));
  }

  frostt = Tensor<double>(prefix + "/vast-2015-mc1-5d.tns", true);
  frostt._infer_dimensionality();
  frostt._infer_shape();
  if (averages_per_tensor.find("vast014") == averages_per_tensor.end()) {
    averages_per_tensor["vast014"] = std::vector<float>();
  }
  for (auto tile_size : tile_sizes) {
    averages_per_tensor["vast014"].push_back(
        frostt.total_active_columns(CoOrdinate({0, 1, 4}), tile_size));
  }
}


std::vector<uint64_t> generate_uniform_random_integers(uint64_t n, int min_val, uint64_t max_val) {
    // Seed the random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::cout<<"n: "<<n<<std::endl;
    std::cout<<"min_val: "<<min_val<<std::endl;
    std::cout<<"max_val: "<<max_val<<std::endl;

    // Define the uniform distribution
    std::uniform_int_distribution<uint64_t> distrib(min_val, max_val);

    // Generate the random numbers
    std::vector<uint64_t> random_numbers(n);
    for (uint64_t i = 0; i < n; ++i) {
        random_numbers[i] = distrib(gen);
    }

    return random_numbers;
}

Tensor<double> make_data(uint64_t leading, uint64_t contraction, float density){
    uint64_t dense_area = leading * contraction;
    uint64_t nnz = dense_area * density;
    Tensor<double> result;
    auto co_ords = generate_uniform_random_integers(nnz, 0, dense_area);
    for(auto co_ord : co_ords){
        CoOrdinate this_cord({static_cast<int>(co_ord/contraction), static_cast<int>(co_ord%contraction)});
        result.get_nonzeros().push_back(NNZ<double>(1.0, this_cord));
    }
    result._infer_dimensionality();
    result._infer_shape();
    //for(uint64_t _iter = 0; _iter < nnz; _iter++){
    //    uint64_t this_pos = std::ceil(_iter/density);
    //    CoOrdinate this_cord({static_cast<int>(this_pos/contraction), static_cast<int>(this_pos%contraction)});
    //    result.get_nonzeros().push_back(NNZ<double>(1.0, this_cord));
    //}
    return result;
}

void run_synthetic_data(int R, float density, std::ofstream &outf, int left_tile_size, int right_tile_size, Tensor<double> &synthetic) {
    std::cout << "Running synthetic tensor shape " << R << ", " << R
              << ". Density " << density << std::endl;
    std::chrono::high_resolution_clock::time_point t1 =
        std::chrono::high_resolution_clock::now();
    ListTensor<double> result =
        synthetic
            .parallel_tile2d_outer_multiply<TileAccumulator<double>, double>(
                synthetic, CoOrdinate({1}), CoOrdinate({0}), left_tile_size, right_tile_size);
    std::chrono::high_resolution_clock::time_point t2 =
        std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "Synthetic tensor mode 1: " << time_span.count() << " seconds."
              << std::endl;
    outf << R << "," << density << "," << left_tile_size << ","
         << right_tile_size << ","
         << "dense," << time_span.count() << std::endl;

    t1 = std::chrono::high_resolution_clock::now();
    result =
        synthetic
            .parallel_tile2d_outer_multiply<TileAccumulatorMap<double>, double>(
                synthetic, CoOrdinate({1}), CoOrdinate({0}), left_tile_size, right_tile_size);
    t2 = std::chrono::high_resolution_clock::now();
    time_span =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    outf << R << "," << density << "," << left_tile_size << ","
         << right_tile_size << ","
         << "sparse," << time_span.count() << std::endl;
    std::cout<<"Synthetic tensor mode 1 sparse: "<<time_span.count()<<" seconds."<<std::endl;
}

void run_dropout_frostt_chicago(Tensor<double>& vast, int left_tile_size, int right_tile_size, std::ofstream &outf) {

  std::cout << "Running chicago tensor" << std::endl;
  if (fork() == 0) {
    vast._infer_dimensionality();
    vast._infer_shape();
    std::cout << "mode 0 contraction" << std::endl;
    std::chrono::high_resolution_clock::time_point t1 =
        std::chrono::high_resolution_clock::now();
    ListTensor<double> result = vast.parallel_tile2d_outer_multiply<TileAccumulator<double>, double>(
        vast, CoOrdinate({0}), CoOrdinate({0}), left_tile_size, right_tile_size);
    std::chrono::high_resolution_clock::time_point t2 =
        std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "chicago_0.3 mode 0: " << time_span.count()
              << " seconds." << std::endl;
    outf << "chicago_0.3_0,"<<left_tile_size<<","<<right_tile_size<<","<<",dense," << time_span.count() << std::endl;
    exit(0);
  } else {
    int stat;
    wait(&stat);
  }
  if (fork() == 0) {
    vast._infer_dimensionality();
    vast._infer_shape();
    std::cout << "mode 0 contraction" << std::endl;
    std::chrono::high_resolution_clock::time_point t1 =
        std::chrono::high_resolution_clock::now();
    ListTensor<double> result = vast.parallel_tile2d_outer_multiply<TileAccumulatorMap<double>, double>(
        vast, CoOrdinate({0}), CoOrdinate({0}), left_tile_size, right_tile_size);
    std::chrono::high_resolution_clock::time_point t2 =
        std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "chicago_0.3 mode 0: " << time_span.count()
              << " seconds." << std::endl;
    outf << "chicago_0.3_0,"<<left_tile_size<<","<<right_tile_size<<","<<",sparse," << time_span.count() << std::endl;
    exit(0);
  } else {
    int stat;
    wait(&stat);
  }
  if (fork() == 0) {
    vast._infer_dimensionality();
    vast._infer_shape();
    std::cout << "mode 0 1 contraction" << std::endl;
    std::chrono::high_resolution_clock::time_point t1 =
        std::chrono::high_resolution_clock::now();
    ListTensor<double> result =
        vast.parallel_tile2d_outer_multiply<TileAccumulator<double>, double>(
            vast, CoOrdinate({0, 1}), CoOrdinate({0, 1}), left_tile_size, right_tile_size);
    std::chrono::high_resolution_clock::time_point t2 =
        std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "chicago_0.3 mode 0, 1: " << time_span.count()
              << " seconds." << std::endl;
    outf << "chicago_0.3_0_1,"<<left_tile_size<<","<<right_tile_size<<","<<",dense," << time_span.count() << std::endl;
    exit(0);
  } else {
    int stat;
    wait(&stat);
  }
  if (fork() == 0) {
    vast._infer_dimensionality();
    vast._infer_shape();
    std::cout << "mode 0 1 contraction" << std::endl;
    std::chrono::high_resolution_clock::time_point t1 =
        std::chrono::high_resolution_clock::now();
    ListTensor<double> result =
        vast.parallel_tile2d_outer_multiply<TileAccumulatorMap<double>, double>(
            vast, CoOrdinate({0, 1}), CoOrdinate({0, 1}), left_tile_size, right_tile_size);
    std::chrono::high_resolution_clock::time_point t2 =
        std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "chicago_0.3 mode 0, 1: " << time_span.count()
              << " seconds." << std::endl;
    outf << "chicago_0.3_0_1,"<<left_tile_size<<","<<right_tile_size<<","<<",sparse," << time_span.count() << std::endl;
    exit(0);
  } else {
    int stat;
    wait(&stat);
  }
}

void run_dropout_frostt(Tensor<double>& vast, int left_tile_size, int right_tile_size, std::ofstream &outf) {

  std::cout << "Running vast-5d tensor" << std::endl;
  if (fork() == 0) {
    vast._infer_dimensionality();
    vast._infer_shape();
    std::cout << "mode 0 1 contraction" << std::endl;
    std::chrono::high_resolution_clock::time_point t1 =
        std::chrono::high_resolution_clock::now();
    ListTensor<double> result = vast.parallel_tile2d_outer_multiply<TileAccumulator<double>, double>(
        vast, CoOrdinate({0, 1}), CoOrdinate({0, 1}), left_tile_size, right_tile_size);
    std::chrono::high_resolution_clock::time_point t2 =
        std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "vast_0.3 mode 0, 1: " << time_span.count()
              << " seconds." << std::endl;
    outf << "vast_0.3_0_1,"<<left_tile_size<<","<<right_tile_size<<","<<",dense," << time_span.count() << std::endl;
    exit(0);
  } else {
    int stat;
    wait(&stat);
  }
  if (fork() == 0) {
    vast._infer_dimensionality();
    vast._infer_shape();
    std::cout << "mode 0 1 contraction" << std::endl;
    std::chrono::high_resolution_clock::time_point t1 =
        std::chrono::high_resolution_clock::now();
    ListTensor<double> result = vast.parallel_tile2d_outer_multiply<TileAccumulatorMap<double>, double>(
        vast, CoOrdinate({0, 1}), CoOrdinate({0, 1}), left_tile_size, right_tile_size);
    std::chrono::high_resolution_clock::time_point t2 =
        std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "vast_0.3 mode 0, 1: " << time_span.count()
              << " seconds." << std::endl;
    outf << "vast_0.3_0_1,"<<left_tile_size<<","<<right_tile_size<<","<<",sparse," << time_span.count() << std::endl;
    exit(0);
  } else {
    int stat;
    wait(&stat);
  }
  if (fork() == 0) {
    vast._infer_dimensionality();
    vast._infer_shape();
    std::cout << "mode 0 1 4 contraction" << std::endl;
    std::chrono::high_resolution_clock::time_point t1 =
        std::chrono::high_resolution_clock::now();
    ListTensor<double> result =
        vast.parallel_tile2d_outer_multiply<TileAccumulator<double>, double>(
            vast, CoOrdinate({0, 1, 4}), CoOrdinate({0, 1, 4}), left_tile_size, right_tile_size);
    std::chrono::high_resolution_clock::time_point t2 =
        std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "vast_0.3 mode 0, 1, 4: " << time_span.count()
              << " seconds." << std::endl;
    outf << "vast_0.3_0_1_4,"<<left_tile_size<<","<<right_tile_size<<","<<",dense," << time_span.count() << std::endl;
    exit(0);
  } else {
    int stat;
    wait(&stat);
  }
  if (fork() == 0) {
    vast._infer_dimensionality();
    vast._infer_shape();
    std::cout << "mode 0 1 4 contraction" << std::endl;
    std::chrono::high_resolution_clock::time_point t1 =
        std::chrono::high_resolution_clock::now();
    ListTensor<double> result =
        vast.parallel_tile2d_outer_multiply<TileAccumulatorMap<double>, double>(
            vast, CoOrdinate({0, 1, 4}), CoOrdinate({0, 1, 4}), left_tile_size, right_tile_size);
    std::chrono::high_resolution_clock::time_point t2 =
        std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "vast_0.3 mode 0, 1, 4: " << time_span.count()
              << " seconds." << std::endl;
    outf << "vast_0.3_0_1_4,"<<left_tile_size<<","<<right_tile_size<<","<<",sparse," << time_span.count() << std::endl;
    exit(0);
  } else {
    int stat;
    wait(&stat);
  }
}

void run_frostt_experiments() {
  struct rusage usage_before, usage_after;
  std::string prefix = "/media/saurabh/New Volume1/ubuntu_downloads/frostt/";
  //Tensor<double> nips = Tensor<double>(prefix+"/nips.tns", true);
  //nips._infer_dimensionality();
  //nips._infer_shape();
  //int l2_size = 16 * 1024 * 1024; //in bytes
  //int num_elts = l2_size / sizeof(double);
  //int tile_size = sqrt(scaling_factor * num_elts);
  int left_tile_size = 256 / scaling_factor;
  int right_tile_size = 256 * scaling_factor;
  //int left_tile_size = scaling_factor;
  //int right_tile_size = -1;
  //
  //std::cout << "Running delicious tensor" << std::endl;
  //if (fork() == 0) {
  //  Tensor<double> delicious4d = Tensor<double>(prefix + "/delicious-4d.tns", true);
  //  delicious4d._infer_dimensionality();
  //  delicious4d._infer_shape();
  //  std::cout << "mode 0, 1 contraction" << std::endl;
  //  std::chrono::high_resolution_clock::time_point t1 =
  //      std::chrono::high_resolution_clock::now();
  //  ListTensor<double> result = delicious4d.parallel_tile2d_outer_multiply<double>(
  //      delicious4d, CoOrdinate({1, 2}), CoOrdinate({1, 2}), left_tile_size,
  //      right_tile_size);
  //  std::chrono::high_resolution_clock::time_point t2 =
  //      std::chrono::high_resolution_clock::now();
  //  std::chrono::duration<double> time_span =
  //      std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
  //  std::cout << "delicious4d mode 0, 2: " << time_span.count() << " seconds."
  //            << std::endl;
  //  exit(0);
  //} else {
  //  int stat;
  //  wait(&stat);
  //}

  //std::cout << "Running nell-2 tensor" << std::endl;
  //if (fork() == 0) {
  //  Tensor<double> nell2 = Tensor<double>(prefix + "/nell-2.tns", true);
  //  nell2._infer_dimensionality();
  //  nell2._infer_shape();
  //  std::cout << "mode 0, 1 contraction" << std::endl;
  //  std::chrono::high_resolution_clock::time_point t1 =
  //      std::chrono::high_resolution_clock::now();
  //  ListTensor<double> result = nell2.parallel_tile2d_outer_multiply<double>(
  //      nell2, CoOrdinate({0, 1}), CoOrdinate({0, 1}), left_tile_size, right_tile_size);
  //  std::chrono::high_resolution_clock::time_point t2 =
  //      std::chrono::high_resolution_clock::now();
  //  std::chrono::duration<double> time_span =
  //      std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
  //  std::cout << "nell2 mode 0, 1: " << time_span.count() << " seconds."
  //            << std::endl;
  //  exit(0);
  //} else {
  //  int stat;
  //  wait(&stat);
  //}

  // nips experiments
  //std::cout << "Running nips tensor" << std::endl;
  //if (fork() == 0) {
  //  Tensor<double> nips = Tensor<double>(prefix + "/nips.tns", true);
  //  nips._infer_dimensionality();
  //  nips._infer_shape();
  //  std::cout << "mode 2 contraction" << std::endl;
  //  std::chrono::high_resolution_clock::time_point t1 =
  //      std::chrono::high_resolution_clock::now();
  //  ListTensor<double> result = nips.tile2d_outer_multiply<double>(
  //      nips, CoOrdinate({2}), CoOrdinate({2}), scaling_factor, 134217728);
  //  std::chrono::high_resolution_clock::time_point t2 =
  //      std::chrono::high_resolution_clock::now();
  //  std::chrono::duration<double> time_span =
  //      std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
  //  std::cout << "NIPS mode 2: " << time_span.count() << " seconds."
  //            << std::endl;
  //  exit(0);
  //} else {
  //  int stat;
  //  wait(&stat);
  //}
  //if (fork() == 0) {
  //  Tensor<double> nips = Tensor<double>(prefix + "/nips.tns", true);
  //  nips._infer_dimensionality();
  //  nips._infer_shape();
  //  std::cout << "mode 2 3 contraction" << std::endl;
  //  std::chrono::high_resolution_clock::time_point t1 =
  //      std::chrono::high_resolution_clock::now();
  //  ListTensor<double> result = nips.tile2d_outer_multiply<double>(
  //      nips, CoOrdinate({2, 3}), CoOrdinate({2, 3}), scaling_factor, 8388608);
  //  std::chrono::high_resolution_clock::time_point t2 =
  //      std::chrono::high_resolution_clock::now();
  //  std::chrono::duration<double> time_span =
  //      std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
  //  std::cout << "NIPS mode 2, 3: " << time_span.count() << " seconds."
  //            << std::endl;
  //  exit(0);
  //} else {
  //  int stat;
  //  wait(&stat);
  //}
  //if (fork() == 0) {
  //  Tensor<double> nips = Tensor<double>(prefix+"/nips.tns", true);
  //  nips._infer_dimensionality();
  //  nips._infer_shape();
  //  std::cout << "mode 0 1 3 contraction" << std::endl;
  //  getrusage(RUSAGE_SELF, &usage_before);
  //  std::chrono::high_resolution_clock::time_point t1 =
  //      std::chrono::high_resolution_clock::now();
  //  ListTensor<double> result = nips.parallel_tile2d_outer_multiply<double>(
  //      nips, CoOrdinate({0, 1, 3}), CoOrdinate({0, 1, 3}), left_tile_size, right_tile_size);
  //  std::chrono::high_resolution_clock::time_point t2 =
  //      std::chrono::high_resolution_clock::now();
  //  getrusage(RUSAGE_SELF, &usage_after);
  //  std::cout << "RAM usage (in KB): "
  //            << usage_after.ru_maxrss - usage_before.ru_maxrss << std::endl;
  //  std::chrono::duration<double> time_span =
  //      std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
  //  std::cout << "NIPS mode 0, 1, 3: " << time_span.count() << " seconds."
  //            << std::endl;
  //  exit(0);
  //} else {
  //  int stat;
  //  wait(&stat);
  //}
  ////std::string prefix = "/media/saurabh/New Volume/ubuntu_downloads/frostt/";

  ////////// chicago experiments
  //std::cout << "Running chicago tensor" << std::endl;
  //if (fork() == 0) {
  //  Tensor<double> chicago =
  //      Tensor<double>(prefix + "/chicago-crime-comm.tns", true);
  //  chicago._infer_dimensionality();
  //  chicago._infer_shape();
  //  std::cout << "mode 0 contraction" << std::endl;
  //  std::chrono::high_resolution_clock::time_point t1 =
  //      std::chrono::high_resolution_clock::now();
  //  ListTensor<double> result = chicago.parallel_tile2d_outer_multiply<double>(
  //      chicago, CoOrdinate({0}), CoOrdinate({0}), left_tile_size, right_tile_size);
  //  std::chrono::high_resolution_clock::time_point t2 =
  //      std::chrono::high_resolution_clock::now();
  //  std::chrono::duration<double> time_span =
  //      std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
  //  std::cout << "chicago mode 0: " << time_span.count() << " seconds."
  //            << std::endl;
  //  exit(0);
  //} else {
  //  int stat;
  //  wait(&stat);
  //}
  //if (fork() == 0) {
  //  Tensor<double> chicago =
  //      Tensor<double>(prefix + "/chicago-crime-comm.tns", true);
  //  chicago._infer_dimensionality();
  //  chicago._infer_shape();
  //  std::cout << "mode 0 1 contraction" << std::endl;
  //  getrusage(RUSAGE_SELF, &usage_before);
  //  std::chrono::high_resolution_clock::time_point t1 =
  //      std::chrono::high_resolution_clock::now();
  //  ListTensor<double> result = chicago.parallel_tile2d_outer_multiply<double>(
  //      chicago, CoOrdinate({0, 1}), CoOrdinate({0, 1}), left_tile_size, right_tile_size);
  //  std::chrono::high_resolution_clock::time_point t2 =
  //      std::chrono::high_resolution_clock::now();
  //  getrusage(RUSAGE_SELF, &usage_after);
  //  std::cout << "RAM usage (in KB): "
  //            << usage_after.ru_maxrss - usage_before.ru_maxrss << std::endl;
  //  std::chrono::duration<double> time_span =
  //      std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
  //  std::cout << "chicago mode 0, 1: " << time_span.count() << " seconds."
  //            << std::endl;
  //  exit(0);
  //} else {
  //  int stat;
  //  wait(&stat);
  //}
  //if (fork() == 0) {
  //  Tensor<double> chicago =
  //      Tensor<double>(prefix + "/chicago-crime-comm.tns", true);
  //  chicago._infer_dimensionality();
  //  chicago._infer_shape();
  //  std::cout << "mode 1 2 3 contraction" << std::endl;
  //  getrusage(RUSAGE_SELF, &usage_before);
  //  std::chrono::high_resolution_clock::time_point t1 =
  //      std::chrono::high_resolution_clock::now();
  //  ListTensor<double> result = chicago.parallel_tile2d_outer_multiply<double>(
  //      chicago, CoOrdinate({1, 2, 3}), CoOrdinate({1, 2, 3}), left_tile_size, right_tile_size);
  //  std::chrono::high_resolution_clock::time_point t2 =
  //      std::chrono::high_resolution_clock::now();
  //  getrusage(RUSAGE_SELF, &usage_after);
  //  std::cout << "RAM usage (in KB): "
  //            << usage_after.ru_maxrss - usage_before.ru_maxrss << std::endl;
  //  std::chrono::duration<double> time_span =
  //      std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
  //  std::cout << "chicago mode 1, 2, 3: " << time_span.count() << " seconds."
  //            << std::endl;
  //  exit(0);
  //} else {
  //  int stat;
  //  wait(&stat);
  //}

  ////////////////// vast-3d experiments
  //std::cout << "Running vast-5d tensor" << std::endl;
  //if (fork() == 0) {
  //  Tensor<double> vast =
  //      Tensor<double>(prefix + "/vast-2015-mc1-5d.tns", true);
  //  vast._infer_dimensionality();
  //  vast._infer_shape();
  //  std::cout << "mode 0 1 contraction" << std::endl;
  //  getrusage(RUSAGE_SELF, &usage_before);
  //  std::chrono::high_resolution_clock::time_point t1 =
  //      std::chrono::high_resolution_clock::now();
  //  ListTensor<double> result = vast.parallel_tile2d_outer_multiply<double>(
  //      vast, CoOrdinate({0, 1}), CoOrdinate({0, 1}), left_tile_size, right_tile_size);
  //  std::chrono::high_resolution_clock::time_point t2 =
  //      std::chrono::high_resolution_clock::now();
  //  getrusage(RUSAGE_SELF, &usage_after);
  //  std::cout << "RAM usage (in GB): "
  //            << (usage_after.ru_maxrss - usage_before.ru_maxrss) << std::endl;
  //  std::chrono::duration<double> time_span =
  //      std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
  //  std::cout << "vast-2015-mc1-5d mode 0, 1: " << time_span.count()
  //            << " seconds." << std::endl;
  //  exit(0);
  //} else {
  //  int stat;
  //  wait(&stat);
  //}
  //if (fork() == 0) {
  //  Tensor<double> vast =
  //      Tensor<double>(prefix + "/vast-2015-mc1-5d.tns", true);
  //  vast._infer_dimensionality();
  //  vast._infer_shape();
  //  std::cout << "mode 0 1 4 contraction" << std::endl;
  //  std::chrono::high_resolution_clock::time_point t1 =
  //      std::chrono::high_resolution_clock::now();
  //  ListTensor<double> result =
  //      vast.parallel_tile2d_outer_multiply<double>(
  //          vast, CoOrdinate({0, 1, 4}), CoOrdinate({0, 1, 4}), left_tile_size, right_tile_size);
  //  std::chrono::high_resolution_clock::time_point t2 =
  //      std::chrono::high_resolution_clock::now();
  //  std::chrono::duration<double> time_span =
  //      std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
  //  std::cout << "vast-2015-mc1-5d mode 0, 1, 4: " << time_span.count()
  //            << " seconds." << std::endl;
  //  exit(0);
  //} else {
  //  int stat;
  //  wait(&stat);
  //}
  ////////
  ///////////////////// uber experiments
  //std::cout << "Running uber tensor" << std::endl;
  //if (fork() == 0) {
  //  Tensor<double> uber = Tensor<double>(prefix + "/uber.tns", true);
  //  uber._infer_dimensionality();
  //  uber._infer_shape();
  //  std::cout << "mode 0 2 contraction" << std::endl;
  //  std::chrono::high_resolution_clock::time_point t1 =
  //      std::chrono::high_resolution_clock::now();
  //  ListTensor<double> result = uber.parallel_tile2d_outer_multiply<double>(
  //      uber, CoOrdinate({0, 2}), CoOrdinate({0, 2}), left_tile_size, right_tile_size);
  //  std::chrono::high_resolution_clock::time_point t2 =
  //      std::chrono::high_resolution_clock::now();
  //  std::chrono::duration<double> time_span =
  //      std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
  //  std::cout << "uber mode 0 2: " << time_span.count() << " seconds."
  //            << std::endl;
  //  exit(0);
  //} else {
  //  int stat;
  //  wait(&stat);
  //}
  //if (fork() == 0) {
  //  Tensor<double> uber = Tensor<double>(prefix + "/uber.tns", true);
  //  uber._infer_dimensionality();
  //  uber._infer_shape();
  //  std::cout << "mode 1 2 3 contraction" << std::endl;
  //  std::chrono::high_resolution_clock::time_point t1 =
  //      std::chrono::high_resolution_clock::now();
  //  ListTensor<double> result = uber.parallel_tile2d_outer_multiply<double>(
  //      uber, CoOrdinate({1, 2, 3}), CoOrdinate({1, 2, 3}), left_tile_size, right_tile_size);
  //  std::chrono::high_resolution_clock::time_point t2 =
  //      std::chrono::high_resolution_clock::now();
  //  std::chrono::duration<double> time_span =
  //      std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
  //  std::cout << "uber mode 1, 2, 3: " << time_span.count() << " seconds."
  //            << std::endl;
  //  exit(0);
  //} else {
  //  int stat;
  //  wait(&stat);
  //}
}

int get_l2_num_elts(){
    int l2_bytes = 2 * 1024 * 1024;
    int l2_numelts = l2_bytes / sizeof(double);
    return l2_numelts;
}

void run_dlpno_experiments() {
    //std::cout << "Running helium data small" << std::endl;
    //int minsize = 100000;
    //int tile_size = sqrt(scaling_factor * minsize);
    //int tile_size = scaling_factor;
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
    //std::cout << "Running helium data large" << std::endl;
    //if (fork() == 0) {
    //    Tensor<double> teov = Tensor<double>(
    //        "./helium_large/threec_int.tns",
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
    //    std::cout << "helium large time : " << time_span.count() << " seconds."
    //              << std::endl;
    //    exit(0);
    //} else {
    //    int stat;
    //    wait(&stat);
    //}
}

Tensor<double> flatten(Tensor<double> input, CoOrdinate leading, CoOrdinate trailing){
    Tensor<double> result;
    for(auto nnz : input.get_nonzeros()){
        CoOrdinate this_cord = nnz.get_coords();
        uint64_t leading_cord = this_cord.gather_linearize(leading);
        uint64_t trailing_cord = this_cord.gather_linearize(trailing);
        CoOrdinate new_cord({static_cast<int>(leading_cord), static_cast<int>(trailing_cord)});
        result.get_nonzeros().push_back(NNZ<double>(nnz.get_data(), new_cord));
    }
    result._infer_dimensionality();
    result._infer_shape();
    return result;
}

void flatten_frostt(std::string prefix){
  std::cout << "Running nips tensor" << std::endl;
  if (fork() == 0) {
    Tensor<double> nips = Tensor<double>(prefix + "/nips.tns", true);
    nips._infer_dimensionality();
    nips._infer_shape();
    std::cout << "mode 2 contraction" << std::endl;
    auto nips_flat = flatten(nips, CoOrdinate({0, 1, 3}), CoOrdinate({2}));
    nips_flat._infer_dimensionality();
    nips_flat._infer_shape();
    nips_flat.write(prefix + "/nips_mode2.tns");
    exit(0);
  } else {
    int stat;
    wait(&stat);
  }
  if (fork() == 0) {
    Tensor<double> nips = Tensor<double>(prefix + "/nips.tns", true);
    nips._infer_dimensionality();
    nips._infer_shape();
    std::cout << "mode 2 3 contraction" << std::endl;
    auto nips_flat = flatten(nips, CoOrdinate({0, 1}), CoOrdinate({2, 3}));
    nips_flat._infer_dimensionality();
    nips_flat._infer_shape();
    nips_flat.write(prefix + "/nips_mode23.tns");
    exit(0);
  } else {
    int stat;
    wait(&stat);
  }
  if (fork() == 0) {
    Tensor<double> nips = Tensor<double>(prefix+"/nips.tns", true);
    nips._infer_dimensionality();
    nips._infer_shape();
    std::cout << "mode 0 1 3 contraction" << std::endl;
    auto nips_flat = flatten(nips, CoOrdinate({2}), CoOrdinate({0, 1, 3}));
    nips_flat._infer_dimensionality();
    nips_flat._infer_shape();
    nips_flat.write(prefix + "/nips_mode013.tns");
    exit(0);
  } else {
    int stat;
    wait(&stat);
  }
  ////std::string prefix = "/media/saurabh/New Volume/ubuntu_downloads/frostt/";

  ////////// chicago experiments
  std::cout << "Running chicago tensor" << std::endl;
  if (fork() == 0) {
    Tensor<double> chicago =
        Tensor<double>(prefix + "/chicago-crime-comm.tns", true);
    chicago._infer_dimensionality();
    chicago._infer_shape();
    std::cout << "mode 0 contraction" << std::endl;
    auto chicago_flat = flatten(chicago, CoOrdinate({1, 2, 3}), CoOrdinate({0}));
    chicago_flat._infer_dimensionality();
    chicago_flat._infer_shape();
    chicago_flat.write(prefix + "/chicago_mode0.tns");
    exit(0);
  } else {
    int stat;
    wait(&stat);
  }
  if (fork() == 0) {
    Tensor<double> chicago =
        Tensor<double>(prefix + "/chicago-crime-comm.tns", true);
    chicago._infer_dimensionality();
    chicago._infer_shape();
    std::cout << "mode 0 1 contraction" << std::endl;
    auto chicago_flat = flatten(chicago, CoOrdinate({2, 3}), CoOrdinate({0, 1}));
    chicago_flat._infer_dimensionality();
    chicago_flat._infer_shape();
    chicago_flat.write(prefix + "/chicago_mode01.tns");
    exit(0);
  } else {
    int stat;
    wait(&stat);
  }
  if (fork() == 0) {
    Tensor<double> chicago =
        Tensor<double>(prefix + "/chicago-crime-comm.tns", true);
    chicago._infer_dimensionality();
    chicago._infer_shape();
    std::cout << "mode 1 2 3 contraction" << std::endl;
    auto chicago_flat = flatten(chicago, CoOrdinate({0}), CoOrdinate({1, 2, 3}));
    chicago_flat._infer_dimensionality();
    chicago_flat._infer_shape();
    chicago_flat.write(prefix + "/chicago_mode123.tns");
    exit(0);
  } else {
    int stat;
    wait(&stat);
  }

  ////////////////// vast-3d experiments
  //std::cout << "Running vast-5d tensor" << std::endl;
  //if (fork() == 0) {
  //  Tensor<double> vast =
  //      Tensor<double>(prefix + "/vast-2015-mc1-5d.tns", true);
  //  vast._infer_dimensionality();
  //  vast._infer_shape();
  //  std::cout << "mode 0 1 contraction" << std::endl;
  //  getrusage(RUSAGE_SELF, &usage_before);
  //  std::chrono::high_resolution_clock::time_point t1 =
  //      std::chrono::high_resolution_clock::now();
  //  ListTensor<double> result = vast.parallel_tile2d_outer_multiply<double>(
  //      vast, CoOrdinate({0, 1}), CoOrdinate({0, 1}), left_tile_size, right_tile_size);
  //  std::chrono::high_resolution_clock::time_point t2 =
  //      std::chrono::high_resolution_clock::now();
  //  getrusage(RUSAGE_SELF, &usage_after);
  //  std::cout << "RAM usage (in GB): "
  //            << (usage_after.ru_maxrss - usage_before.ru_maxrss) << std::endl;
  //  std::chrono::duration<double> time_span =
  //      std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
  //  std::cout << "vast-2015-mc1-5d mode 0, 1: " << time_span.count()
  //            << " seconds." << std::endl;
  //  exit(0);
  //} else {
  //  int stat;
  //  wait(&stat);
  //}
  //if (fork() == 0) {
  //  Tensor<double> vast =
  //      Tensor<double>(prefix + "/vast-2015-mc1-5d.tns", true);
  //  vast._infer_dimensionality();
  //  vast._infer_shape();
  //  std::cout << "mode 0 1 4 contraction" << std::endl;
  //  std::chrono::high_resolution_clock::time_point t1 =
  //      std::chrono::high_resolution_clock::now();
  //  ListTensor<double> result =
  //      vast.parallel_tile2d_outer_multiply<double>(
  //          vast, CoOrdinate({0, 1, 4}), CoOrdinate({0, 1, 4}), left_tile_size, right_tile_size);
  //  std::chrono::high_resolution_clock::time_point t2 =
  //      std::chrono::high_resolution_clock::now();
  //  std::chrono::duration<double> time_span =
  //      std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
  //  std::cout << "vast-2015-mc1-5d mode 0, 1, 4: " << time_span.count()
  //            << " seconds." << std::endl;
  //  exit(0);
  //} else {
  //  int stat;
  //  wait(&stat);
  //}
  ////////
  ///////////////////// uber experiments
  //std::cout << "Running uber tensor" << std::endl;
  //if (fork() == 0) {
  //  Tensor<double> uber = Tensor<double>(prefix + "/uber.tns", true);
  //  uber._infer_dimensionality();
  //  uber._infer_shape();
  //  std::cout << "mode 0 2 contraction" << std::endl;
  //  std::chrono::high_resolution_clock::time_point t1 =
  //      std::chrono::high_resolution_clock::now();
  //  ListTensor<double> result = uber.parallel_tile2d_outer_multiply<double>(
  //      uber, CoOrdinate({0, 2}), CoOrdinate({0, 2}), left_tile_size, right_tile_size);
  //  std::chrono::high_resolution_clock::time_point t2 =
  //      std::chrono::high_resolution_clock::now();
  //  std::chrono::duration<double> time_span =
  //      std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
  //  std::cout << "uber mode 0 2: " << time_span.count() << " seconds."
  //            << std::endl;
  //  exit(0);
  //} else {
  //  int stat;
  //  wait(&stat);
  //}
  //if (fork() == 0) {
  //  Tensor<double> uber = Tensor<double>(prefix + "/uber.tns", true);
  //  uber._infer_dimensionality();
  //  uber._infer_shape();
  //  std::cout << "mode 1 2 3 contraction" << std::endl;
  //  std::chrono::high_resolution_clock::time_point t1 =
  //      std::chrono::high_resolution_clock::now();
  //  ListTensor<double> result = uber.parallel_tile2d_outer_multiply<double>(
  //      uber, CoOrdinate({1, 2, 3}), CoOrdinate({1, 2, 3}), left_tile_size, right_tile_size);
  //  std::chrono::high_resolution_clock::time_point t2 =
  //      std::chrono::high_resolution_clock::now();
  //  std::chrono::duration<double> time_span =
  //      std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
  //  std::cout << "uber mode 1, 2, 3: " << time_span.count() << " seconds."
  //            << std::endl;
  //  exit(0);
  //} else {
  //  int stat;
  //  wait(&stat);
  //}

}

void run_chemistry_experiments(std::string prefix) {
    int left_tile_size = 256/scaling_factor;
    int right_tile_size = 256 * scaling_factor;
    Tensor<double> teov = Tensor<double>(prefix+"/TEov.tns", true);
    teov._infer_dimensionality();
    teov._infer_shape();
    Tensor<double> teoo = Tensor<double>(prefix+"/TEoo.tns", true);
    teoo._infer_dimensionality();
    teoo._infer_shape();
    //if (fork() == 0) {
    //    std::cout << "Time for TEov * TEov " << std::endl;
    //    std::chrono::high_resolution_clock::time_point t1 =
    //        std::chrono::high_resolution_clock::now();
    //    ListTensor<double> result = teov.tile2d_outer_multiply<double>(
    //        teov, CoOrdinate({0}), CoOrdinate({0}), left_tile_size, right_tile_size);
    //    std::chrono::high_resolution_clock::time_point t2 =
    //        std::chrono::high_resolution_clock::now();
    //    std::chrono::duration<double> time_span =
    //        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    //    std::cout << "ovov time : " << time_span.count() << " seconds."
    //              << std::endl;

    //    exit(0);
    //} else {
    //    int stat;
    //    wait(&stat);
    //}
    //if (fork() == 0) {
    //    Tensor<double> tevv = Tensor<double>(prefix+"/TEvv.tns", true);
    //    tevv._infer_dimensionality();
    //    tevv._infer_shape();
    //    std::cout << "Time for TEvv * TEoo " << std::endl;
    //    std::chrono::high_resolution_clock::time_point t1 =
    //        std::chrono::high_resolution_clock::now();
    //    ListTensor<double> result = tevv.parallel_tile2d_outer_multiply<double>(
    //        teoo, CoOrdinate({2}), CoOrdinate({2}), left_tile_size,
    //        right_tile_size);
    //    std::chrono::high_resolution_clock::time_point t2 =
    //        std::chrono::high_resolution_clock::now();
    //    std::chrono::duration<double> time_span =
    //        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    //    std::cout << "vvoo time : " << time_span.count() << " seconds."
    //              << std::endl;
    //    exit(0);
    //} else {
    //    int stat;
    //    wait(&stat);
    //}
    //if (fork() == 0) {
    //    Tensor<double> tevv = Tensor<double>(prefix+"/TEvv.tns", true);
    //    tevv._infer_dimensionality();
    //    tevv._infer_shape();
    //    std::cout << "Time for TEvv * TEov " << std::endl;
    //    std::chrono::high_resolution_clock::time_point t1 =
    //        std::chrono::high_resolution_clock::now();
    //    ListTensor<double> result = tevv.parallel_tile2d_outer_multiply<double>(
    //        teov, CoOrdinate({2}), CoOrdinate({2}), left_tile_size,
    //        right_tile_size);
    //    std::chrono::high_resolution_clock::time_point t2 =
    //        std::chrono::high_resolution_clock::now();
    //    std::chrono::duration<double> time_span =
    //        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    //    std::cout << "vvov time : " << time_span.count() << " seconds."
    //              << std::endl;
    //    exit(0);
    //} else {
    //    int stat;
    //    wait(&stat);
    //}
}

int main() {

    //flatten_frostt("/media/saurabh/New Volume1/ubuntu_downloads/frostt");
  //std::vector<float> tile_scaling_factors = {0.04, 0.2, 1, 5, 25, 125, 625, 78125, 390625};
  std::vector<int> tile_scaling_factors = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};
  //std::vector<float> tile_scaling_factors = {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};
  //std::vector<float> tile_scaling_factors = {1, 2};
  //std::vector<float> tile_scaling_factors = {43.2, 12.5, 6.25};
  //std::vector<float> tile_scaling_factors = {65536, 131072, 262144, 524288, 1048576, 2097512, 4194304};
  //std::vector<float> tile_scaling_factors = {4096, 8192, 16384, 32678};

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

  //std::cout<<"Guanine data"<<std::endl;
  //Tensor<double> tevv = Tensor<double>("./guanine_data/TEvv.tns", true);
  //tevv._infer_dimensionality();
  //tevv._infer_shape();
  //Tensor<double> teov = Tensor<double>("./guanine_data/TEov.tns", true);
  //teov._infer_dimensionality();
  //teov._infer_shape();
  //Tensor<double> teoo = Tensor<double>("./guanine_data/TEoo.tns", true);
  //teoo._infer_dimensionality();
  //teoo._infer_shape();
  //run_frostt_experiments();

    //std::ofstream file;
    //file.open("synthetic_vast.csv");
    //file << "R,density,left_tile_size, right_tile_size,acc_type,time"
    //     << std::endl;
    //for (int R = 65536; R <= 65536; R*=2) {
    //for (float density = 1.0 / (1 << 8); density <= 1.0 / (1 << 5);
    //     density *= 2) {
      //Tensor<double> synthetic = make_data(R, R, density);
      //int R = 32768;
      //float density = 0.015625;
      //Tensor<double> synthetic = make_data(R, R, density);
      //synthetic._infer_dimensionality();
      //synthetic._infer_shape();
      //for (int scaling = 1; scaling <= 256; scaling *= 2) {
      //  int left_tile_size = 256 / scaling;
      //  int right_tile_size = 256 * scaling;
        //int left_tile_size = 4;
        //int right_tile_size = 16384;
        //if (fork() == 0) {
        //  //run_synthetic_data(R, density, file, left_tile_size, right_tile_size,
        //  //                   synthetic);
        //  run_synthetic_data(R, density, file, left_tile_size, right_tile_size,
        //                     synthetic);
        //  exit(0);
        //} else {
        //  wait(nullptr);
        //}
      //}
      //for (int left_tile_size = 1; left_tile_size <= 256; left_tile_size *= 2) {
      //  if (fork() == 0) {
      //    run_synthetic_data(R, density, file, left_tile_size, R, synthetic);
      //    exit(0);
      //  } else {
      //    wait(nullptr);
      //  }
      //}
    //}
    //}
    //file.close();
        //run_dlpno_experiments();
        //std::cout<<"Caffeine data"<<std::endl;
        //run_chemistry_experiments("./caffeine_data");
        //run_frostt_experiments();
    make_frostt_signatures(tile_scaling_factors);
    std::ofstream file;
    file.open("frostt_signatures.csv");
    file<<"tensor_mode,";
    for(auto item: tile_scaling_factors){
        file<<item<<",";
    }
    file<<std::endl;
    for(auto item: averages_per_tensor){
        file<<item.first<<",";
        for(auto val: item.second){
            file<<val<<",";
        }
        file<<std::endl;
    }
    //file<<"tensor_mode,left_tile_size,right_tile_size,accumulator_type,time"<<std::endl;
    //Tensor<double> vast_red = Tensor<double>("/media/saurabh/New Volume1/ubuntu_downloads/frostt/chicago_0.300000.tns", true);
    //vast_red._infer_dimensionality();
    //vast_red._infer_shape();
    //for(int scaling = 1; scaling <= 256; scaling *= 2){
    //    std::cout << "Scaling factor: " << scaling_factor << std::endl;
    //    run_dropout_frostt_chicago(vast_red, 256/scaling, 256*scaling, file);
    //    run_dropout_frostt_chicago(vast_red, scaling, -1, file);
    //}
}
