#include "sparse_opcount/contract.hpp"
#include "sparse_opcount/read.hpp"
#include <chrono>
#include <iostream>
#include <sys/resource.h>
#include <sys/wait.h>
#include <vector>
static float scaling_factor = 1;
std::map<std::string, std::vector<float>> averages_per_tensor;

template <typename TileType>
double run_tile_experiment(std::string filename, CoOrdinate l_dims, CoOrdinate r_dims, int l_tile_size, int r_tile_size) {

  std::string prefix = "/home/hunter/work/frostt/";
  //std::string prefix = "/media/saurabh/New Volume1/ubuntu_downloads/frostt/";
  //Tensor<double> nips = Tensor<double>(prefix+"/nips.tns", true);
  //nips._infer_dimensionality();
  //nips._infer_shape();
  //int l2_size = 16 * 1024 * 1024; //in bytes
  //int num_elts = l2_size / sizeof(double);
  //int tile_size = sqrt(scaling_factor * num_elts);
  //int left_tile_size = scaling_factor;
  //int right_tile_size = -1;
  //
  std::cout << "Running tensor " << filename << std::endl;

  //start pipe.
  int pipefd[2];
  if (pipe(pipefd) == -1){
    std::cerr << "Pipe creation failed\n";
  }
  if (fork() == 0) {
   close(pipefd[0]);
   Tensor<double> delicious4d = Tensor<double>(prefix + filename, true);
   delicious4d._infer_dimensionality();
   delicious4d._infer_shape();
   std::chrono::high_resolution_clock::time_point t1 =
       std::chrono::high_resolution_clock::now();
   ListTensor<double> result = delicious4d.parallel_tile2d_outer_multiply<TileType, double>(
       delicious4d, l_dims, r_dims, l_tile_size,
       r_tile_size);
      // ListTensor<double> result = delicious4d.parallel_tile2d_outer_multiply<double>(
   //     delicious4d, CoOrdinate({1, 2}), CoOrdinate({1, 2}), left_tile_size,
   //     right_tile_size);
   std::chrono::high_resolution_clock::time_point t2 =
       std::chrono::high_resolution_clock::now();
   std::chrono::duration<double> time_span =
       std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);


   double elapsed = time_span.count();
   write(pipefd[1], &elapsed, sizeof(double));
   close(pipefd[1]);

   exit(0);
  } else {

   close(pipefd[1]);
   int stat;
   wait(&stat);
   double return_val;
   read(pipefd[0], &return_val, sizeof(double));
   close(pipefd[0]);
   return return_val;

  }

}

int get_l2_num_elts(){
    int l2_bytes = 2 * 1024 * 1024;
    int l2_numelts = l2_bytes / sizeof(double);
    return l2_numelts;
}


void run_experiment(std::string filename, CoOrdinate contraction_coord, int tile_size){

  double dense_time = run_tile_experiment<TileAccumulator<double>>(filename, contraction_coord, contraction_coord, tile_size, tile_size);
  double mask_8_time = run_tile_experiment<maskedAccumulator<double, uint8_t>>(filename, contraction_coord, contraction_coord, tile_size, tile_size);
  double mask_16_time = run_tile_experiment<maskedAccumulator<double, uint16_t>>(filename, contraction_coord, contraction_coord, tile_size, tile_size);
  double mask_32_time = run_tile_experiment<maskedAccumulator<double, uint32_t>>(filename, contraction_coord, contraction_coord, tile_size, tile_size);
  double mask_64_time = run_tile_experiment<maskedAccumulator<double, uint64_t>>(filename, contraction_coord, contraction_coord, tile_size, tile_size);

  std::cout << "Times for " << filename << " " << tile_size << std::endl;
  std::cout << dense_time << " " << mask_8_time << " " << mask_16_time << " " << mask_32_time << " " << mask_64_time << std::endl;
}



void run_all_tile_sizes(std::string filename, CoOrdinate contraction_coord){

  run_experiment(filename, contraction_coord, 64);
  run_experiment(filename, contraction_coord, 128);
  run_experiment(filename, contraction_coord, 256);
  run_experiment(filename, contraction_coord, 512);
  run_experiment(filename, contraction_coord, 1024);

}

int main() {

    //flatten_frostt("/media/saurabh/New Volume1/ubuntu_downloads/frostt");
  //std::vector<float> tile_scaling_factors = {0.04, 0.2, 1, 5, 25, 125, 625, 78125, 390625};
  //std::vector<int> tile_scaling_factors = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};
 

  run_all_tile_sizes("nips.tns", CoOrdinate({0,1,3}));

  run_all_tile_sizes("chicago.tns", CoOrdinate({0}));
  run_all_tile_sizes("chicago.tns", CoOrdinate({0,1}));
  run_all_tile_sizes("chicago.tns", CoOrdinate({1,2,3}));

  run_all_tile_sizes("vast-5d.tns", CoOrdinate({0,1}));
  run_all_tile_sizes("vast-5d.tns", CoOrdinate({0,1,4}));

  run_all_tile_sizes("uber.tns", CoOrdinate({0,2}));
  run_all_tile_sizes("uber.tns", CoOrdinate({1,2,3}));

  // double dense_time = run_tile_experiment<TileAccumulatorDense<double>>(filename, CoOrdinate({0}), CoOrdinate({0}), 128, 128);
  // double mask_8_time = run_tile_experiment<maskedAccumulator<double, uint8_t>>(filename, CoOrdinate({0}), CoOrdinate({0}), 128, 128);
  // double mask_16_time = run_tile_experiment<maskedAccumulator<double, uint16_t>>(filename, CoOrdinate({0}), CoOrdinate({0}), 128, 128);
  // double mask_32_time = run_tile_experiment<maskedAccumulator<double, uint32_t>>(filename, CoOrdinate({0}), CoOrdinate({0}), 128, 128);
  // double mask_64_time = run_tile_experiment<maskedAccumulator<double, uint64_t>>(filename, CoOrdinate({0}), CoOrdinate({0}), 128, 128);
  // std::string filename = "uber.tns";
  // double dense_time = run_tile_experiment<TileAccumulatorDense<double>>(filename, CoOrdinate({0,3}), CoOrdinate({0,3}), 128, 128);
  // double mask_8_time = run_tile_experiment<maskedAccumulator<double, uint8_t>>(filename, CoOrdinate({0,3}), CoOrdinate({0,3}), 128, 128);
  // double mask_16_time = run_tile_experiment<maskedAccumulator<double, uint16_t>>(filename, CoOrdinate({0,3}), CoOrdinate({0,3}), 128, 128);
  // double mask_32_time = run_tile_experiment<maskedAccumulator<double, uint32_t>>(filename, CoOrdinate({0,3}), CoOrdinate({0,3}), 128, 128);
  // double mask_64_time = run_tile_experiment<maskedAccumulator<double, uint64_t>>(filename, CoOrdinate({0,3}), CoOrdinate({0,3}), 128, 128);
}
