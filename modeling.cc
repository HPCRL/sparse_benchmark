#include "sparse_opcount/contract.hpp"
#include "sparse_opcount/read.hpp"
#include <chrono>
#include <iostream>
#include <sys/resource.h>
#include <sys/wait.h>
#include <vector>
static float scaling_factor = 1;

void make_a_run(Tensor<double> &some_tensor, std::string exp_name,
                CoOrdinate contr, int left_tile_size, int right_tile_size,
                std::ofstream &out) {
  if (fork() == 0) {
    some_tensor._infer_dimensionality();
    some_tensor._infer_shape();
    std::chrono::high_resolution_clock::time_point t1 =
        std::chrono::high_resolution_clock::now();
    auto [q, dv] = some_tensor.compute_cost(some_tensor, contr, contr,
                                            left_tile_size, right_tile_size);
    std::chrono::high_resolution_clock::time_point t2 =
        std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    out << exp_name << "," << left_tile_size << "," << right_tile_size << ","
        << "dense"
        << "," << q << "," << dv << std::endl;
    exit(0);
  } else {
    int stat;
    wait(&stat);
  }
}

void run_frostt_experiments(Tensor<double> &tensor, std::string tname,
                            int left_tile_size, int right_tile_size,
                            std::ofstream &out) {

  if (tname == "nips.tns") {

    // nips experiments
    std::cout << "Running nips tensor" << std::endl;
    make_a_run(tensor, "NIPS 2", CoOrdinate({2}), left_tile_size,
               right_tile_size, out);
    make_a_run(tensor, "NIPS 2 3", CoOrdinate({2, 3}), left_tile_size,
               right_tile_size, out);
    make_a_run(tensor, "NIPS 0 1 3", CoOrdinate({0, 1, 3}), left_tile_size,
               right_tile_size, out);
  } else if (tname == "chicago-crime-comm.tns") {
    ////////// chicago experiments
    std::cout << "Running chicago tensor" << std::endl;
    make_a_run(tensor, "Chicago 0", CoOrdinate({0}), left_tile_size,
               right_tile_size, out);
    make_a_run(tensor, "Chicago 0 1", CoOrdinate({0, 1}), left_tile_size,
               right_tile_size, out);
    make_a_run(tensor, "Chicago 1 2 3", CoOrdinate({1, 2, 3}), left_tile_size,
               right_tile_size, out);
  } else if (tname == "vast-2015-mc1-5d.tns") {
    ////////////////// vast-3d experiments
    std::cout << "Running vast-5d tensor" << std::endl;
    make_a_run(tensor, "Vast 0 1", CoOrdinate({0, 1}), left_tile_size,
               right_tile_size, out);
    make_a_run(tensor, "Vast 0 1 4", CoOrdinate({0, 1, 4}), left_tile_size,
               right_tile_size, out);
  } else if (tname == "uber.tns") {
    /////////////////// uber experiments
    std::cout << "Running uber tensor" << std::endl;
    make_a_run(tensor, "Uber 0 2", CoOrdinate({0, 2}), left_tile_size,
               right_tile_size, out);
    make_a_run(tensor, "Uber 1 2 3", CoOrdinate({1, 2, 3}), left_tile_size,
               right_tile_size, out);
  }
}

int main() {
  std::ofstream file;
  file.open("frostt_costmodel2.csv");
  file << "Tensor mode,left_tile_size, "
          "right_tile_size,acc_type,queries,data_volume"
       << std::endl;
  std::vector<std::string> tensor_names = {"nips.tns", "chicago-crime-comm.tns",
                                           "vast-2015-mc1-5d.tns", "uber.tns"};

  for (auto tname : tensor_names) {
    Tensor<double> t = Tensor<double>(
        "/media/saurabh/New Volume1/ubuntu_downloads/frostt/" + tname, true);
    for (int base_dim = 256; base_dim <= 1024; base_dim *= 2) {
      for (int scaling = 1; scaling <= base_dim; scaling *= 2) {
        run_frostt_experiments(t, tname, base_dim / scaling, base_dim * scaling,
                               file);
      }
    }
  }

  file.close();
}
