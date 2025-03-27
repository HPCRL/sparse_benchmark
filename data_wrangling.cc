#include "sparse_opcount/contract.hpp"
#include "sparse_opcount/read.hpp"
#include <chrono>
#include <iostream>
#include <sys/resource.h>
#include <sys/wait.h>
#include <vector>
static float scaling_factor = 1;

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

Tensor<double> dropout(Tensor<double>& original, float drop_percentage){
    uint64_t new_num_nnz = original.get_nonzeros().size() * (1 - drop_percentage);
    Tensor<double> result;
    auto selected_pos = generate_uniform_random_integers(new_num_nnz, 0, original.get_nonzeros().size());
    for(auto pos : selected_pos){
        result.get_nonzeros().push_back(original.get_nonzeros()[pos]);
    }
    return result;
}

void dropout_frostt(std::string prefix) {
    //std::cout << "Running vast tensor" << std::endl;
    float drop_percentage = 0.3;
    //if (fork() == 0) {
        //Tensor<double> vast =
        //    Tensor<double>(prefix + "/vast-2015-mc1-5d.tns", true);
        //vast._infer_dimensionality();
        //vast._infer_shape();
        //Tensor<double> vast_dropout = dropout(vast, drop_percentage);
        //vast_dropout._infer_dimensionality();
        //vast_dropout._infer_shape();
        //vast_dropout.write(prefix + "/vast_" + std::to_string(drop_percentage) +
        //                   ".tns");
    //    exit(0);
    //} else {
    //    int stat;
    //    wait(&stat);
    //}
    std::cout << "Running chicago tensor" << std::endl;
        Tensor<double> chicago =
            Tensor<double>(prefix + "/chicago-crime-comm.tns", true);
        chicago._infer_dimensionality();
        chicago._infer_shape();
        Tensor<double> chicago_dropout = dropout(chicago, drop_percentage);
        chicago_dropout._infer_dimensionality();
        chicago_dropout._infer_shape();
        chicago_dropout.write(prefix + "/chicago_" +
                              std::to_string(drop_percentage) + ".tns");
    
}

int main() {
    std::string prefix = "/media/saurabh/New Volume1/ubuntu_downloads/frostt";
    dropout_frostt(prefix);
}
