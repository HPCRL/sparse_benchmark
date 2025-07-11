#include "sparse_opcount/contract.hpp"
#include "sparse_opcount/read.hpp"
#include <chrono>
#include <iostream>
#include <sys/resource.h>
#include <sys/wait.h>
#include <vector>

double make_a_run(Tensor<double> &some_tensor, std::string exp_name,
                CoOrdinate contr, int tile_size, bool dense) {
  int pipefd[2]; // pipefd[0] for reading, pipefd[1] for writing
  if (pipe(pipefd) == -1) {
    perror("pipe");
    return -1;
  }
  if (fork() == 0) {
    some_tensor._infer_dimensionality();
    some_tensor._infer_shape();
    std::chrono::high_resolution_clock::time_point t1 =
        std::chrono::high_resolution_clock::now();
    if (dense) {
      ListTensor<double> result =
          some_tensor.fastcc_multiply<TileAccumulator<double>, double, double>(
              some_tensor, contr, contr, tile_size);
    } else {
      ListTensor<double> result =
          some_tensor
              .fastcc_multiply<TileAccumulatorMap<double>, double, double>(
                  some_tensor, contr, contr, tile_size);
    }
    std::chrono::high_resolution_clock::time_point t2 =
        std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    close(pipefd[0]); // Close unused read end in child
    write(pipefd[1], &time_span, sizeof(time_span));
    close(pipefd[1]); // Close write end
    exit(0);
  } else {
    int stat;
    double received_value = -1.0;
    close(pipefd[1]); // Close unused write end in parent
    // Wait for the child process to terminate and get its exit status
    if (wait(&stat) == -1) {
      perror("waitpid");
      return -1.0;
    }
    if (WIFEXITED(stat)) {
      int exitStatus = WEXITSTATUS(stat);
      if (exitStatus == EXIT_SUCCESS) {
        // Child exited normally, try to read from the pipe
        if (read(pipefd[0], &received_value, sizeof(double)) == -1) {
          perror("parent read");
          exit(EXIT_FAILURE);
        }
        std::cout << "Parent received: " << received_value << std::endl;
      } else {
        std::cerr << "Parent: Child process failed with exit status "
                  << exitStatus << std::endl;
        return -1.0;
      }
    } else if (WIFSIGNALED(stat)) {
      std::cerr << "Parent: Child process terminated by signal "
                << WTERMSIG(stat) << std::endl;
      // Handle the child's termination by signal appropriately
      return -1.0;
    }
    close(pipefd[0]); // Close read end
    return received_value;
  }
}

double run_a_times_b(Tensor<double>& a, Tensor<double>& b, CoOrdinate contr, int tile_size, bool dense){
    int pipefd[2]; // pipefd[0] for reading, pipefd[1] for writing
    if (pipe(pipefd) == -1) {
        perror("pipe");
        return -1;
    }
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
        close(pipefd[0]); // Close unused read end in child
        write(pipefd[1], &time_span, sizeof(time_span));
        close(pipefd[1]); // Close write end
        exit(0); // Exit the child process
    } else {
        int stat;
        double received_value = -1.0;
        close(pipefd[1]); // Close unused write end in parent
        // Wait for the child process to terminate and get its exit status
        if (wait(&stat) == -1) {
            perror("waitpid");
            return -1.0;
        }
        if (WIFEXITED(stat)) {
            int exitStatus = WEXITSTATUS(stat);
            if (exitStatus == EXIT_SUCCESS) {
                // Child exited normally, try to read from the pipe
                if (read(pipefd[0], &received_value, sizeof(double)) == -1) {
                    perror("parent read");
                    exit(EXIT_FAILURE);
                }
                std::cout << "Parent received: " << received_value << std::endl;
            } else {
                std::cerr << "Parent: Child process failed with exit status " << exitStatus << std::endl;
                return -1.0;
            }
        } else if (WIFSIGNALED(stat)) {
            std::cerr << "Parent: Child process terminated by signal " << WTERMSIG(stat) << std::endl;
            // Handle the child's termination by signal appropriately
            return -1.0;
        }
        close(pipefd[0]); // Close read end
        return received_value;
    }
}

void run_frostt_experiments(std::vector<int> tile_sizes, std::ostream& out, std::string& prefix) {

  double minimum_times[10] = {1<<30, 1<<30, 1<<30, 1<<30, 1<<30, 1<<30, 1<<30, 1<<30, 1<<30, 1<<30};
  Tensor<double> nips = Tensor<double>(prefix + "/nips.tns", true);
  Tensor<double> chicago = Tensor<double>(prefix + "/chicago-crime-comm.tns", true);
  Tensor<double> vast = Tensor<double>(prefix + "/vast-2015-mc1-5d.tns", true);
  Tensor<double> uber = Tensor<double>(prefix + "/uber.tns", true);
  for (auto s : tile_sizes) {
        double time_taken = 0.0;
        // nips experiments
        std::cout << "Running nips tensor" << std::endl;
        time_taken = make_a_run(nips, "NIPS 2", CoOrdinate({2}), s * 1024, false);
        std::cout<<"Time taken for NIPS 2: " << time_taken << " seconds at tile size " << s * 1024 << std::endl;
        minimum_times[0] = time_taken != -1.0 ? std::min(minimum_times[0], time_taken) : minimum_times[0];
        time_taken = make_a_run(nips, "NIPS 2 3", CoOrdinate({2, 3}), s * 1024, false);
        std::cout<<"Time taken for NIPS 2 3: " << time_taken << " seconds at tile size " << s * 1024 << std::endl;
        minimum_times[1] = time_taken != -1.0 ? std::min(minimum_times[1], time_taken) : minimum_times[1];
        time_taken = make_a_run(nips, "NIPS 0 1 3", CoOrdinate({0, 1, 3}), s, true);
        std::cout<<"Time taken for NIPS 0 1 3: " << time_taken << " seconds at tile size " << s << std::endl;
        minimum_times[2] = time_taken != -1.0 ? std::min(minimum_times[2], time_taken) : minimum_times[2];

        ////////////// chicago experiments
        std::cout << "Running chicago tensor" << std::endl;
        time_taken = make_a_run(chicago, "Chicago 0", CoOrdinate({0}), s, true);
        std::cout<<"Time taken for Chicago 0: " << time_taken << " seconds at tile size " << s << std::endl;
        minimum_times[3] = time_taken != -1.0 ? std::min(minimum_times[3], time_taken) : minimum_times[3];
        time_taken = make_a_run(chicago, "Chicago 0 1", CoOrdinate({0, 1}), s, true);
        std::cout<<"Time taken for Chicago 0 1: " << time_taken << " seconds at tile size " << s << std::endl;
        minimum_times[4] = time_taken != -1.0 ? std::min(minimum_times[4], time_taken) : minimum_times[4];
        time_taken = make_a_run(chicago, "Chicago 1 2 3", CoOrdinate({1, 2, 3}), s, true);
        std::cout<<"Time taken for Chicago 1 2 3: " << time_taken << " seconds at tile size " << s << std::endl;
        minimum_times[5] = time_taken != -1.0 ? std::min(minimum_times[5], time_taken) : minimum_times[5];

        //////////////// vast-3d experiments
        std::cout << "Running vast-5d tensor" << std::endl;
        time_taken = make_a_run(vast, "Vast 0 1", CoOrdinate({0, 1}), s, true);
        std::cout<<"Time taken for Vast 0 1: " << time_taken << " seconds at tile size " << s << std::endl;
        minimum_times[6] = time_taken != -1.0 ? std::min(minimum_times[6], time_taken) : minimum_times[6];
        time_taken = make_a_run(vast, "Vast 0 1 4", CoOrdinate({0, 1, 4}), s, true);
        std::cout<<"Time taken for Vast 0 1 4: " << time_taken << " seconds at tile size " << s << std::endl;
        minimum_times[7] = time_taken != -1.0 ? std::min(minimum_times[7], time_taken) : minimum_times[7];

        /////////////////// uber experiments
        std::cout << "Running uber tensor" << std::endl;
        time_taken = make_a_run(uber, "Uber 0 2", CoOrdinate({0, 2}), s, true);
        std::cout<<"Time taken for Uber 0 2: " << time_taken << " seconds at tile size " << s << std::endl;
        minimum_times[8] = time_taken != -1.0 ? std::min(minimum_times[8], time_taken) : minimum_times[8];
        time_taken = make_a_run(uber, "Uber 1 2 3", CoOrdinate({1, 2, 3}), s, true);
        std::cout<<"Time taken for Uber 1 2 3: " << time_taken << " seconds at tile size " << s << std::endl;
        minimum_times[9] = time_taken != -1.0 ? std::min(minimum_times[9], time_taken) : minimum_times[9];
  }
  out << "NIPS 2"
      << "," << minimum_times[0] << std::endl;
  out << "NIPS 2 3"
      << "," << minimum_times[1] << std::endl;
  out << "NIPS 0 1 3"
      << "," << minimum_times[2] << std::endl;
  out << "Chicago 0"
      << "," << minimum_times[3] << std::endl;
  out << "Chicago 0 1"
      << "," << minimum_times[4] << std::endl;
  out << "Chicago 1 2 3"
      << "," << minimum_times[5] << std::endl;
  out << "Vast 0 1"
      << "," << minimum_times[6] << std::endl;
  out << "Vast 0 1 4"
      << "," << minimum_times[7] << std::endl;
  out << "Uber 0 2"
      << "," << minimum_times[8] << std::endl;
  out << "Uber 1 2 3"
      << "," << minimum_times[9] << std::endl;
}

void run_chemistry_experiments(std::vector<int> tile_sizes, std::ostream &out,
                               std::string &caffeine_prefix,
                               std::string &guanine_prefix) {
  double minimum_times[6] = {1<<30, 1<<30, 1<<30, 1<<30, 1<<30, 1<<30};

  Tensor<double> tevv_caffeine =
      Tensor<double>(caffeine_prefix + "/TEvv.tns", true);
  Tensor<double> teoo_caffeine =
      Tensor<double>(caffeine_prefix + "/TEoo.tns", true);
  Tensor<double> teov_caffeine =
      Tensor<double>(caffeine_prefix + "/TEov.tns", true);
  Tensor<double> tevv_guanine =
      Tensor<double>(guanine_prefix + "/TEvv.tns", true);
  Tensor<double> teoo_guanine =
      Tensor<double>(guanine_prefix + "/TEoo.tns", true);
  Tensor<double> teov_guanine =
      Tensor<double>(guanine_prefix + "/TEov.tns", true);
  for (auto s : tile_sizes) {
        double time_taken = 0.0;
        ///////////// caffeine experiments
        std::cout << "Running caffeine VVOO" << std::endl;
        time_taken = run_a_times_b(tevv_caffeine, teoo_caffeine,
                                   CoOrdinate({2}), s, true);
        std::cout << "Time taken for caffeine VVOO: " << time_taken
                  << " seconds at tile size " << s << std::endl;
        minimum_times[0] = time_taken != -1.0
                               ? std::min(minimum_times[0], time_taken)
                               : minimum_times[0];
        time_taken = run_a_times_b(teov_caffeine, teov_caffeine,
                                   CoOrdinate({2}), s, true);
        std::cout << "Time taken for caffeine OVOV: " << time_taken
                  << " seconds at tile size " << s << std::endl;
        minimum_times[1] = time_taken != -1.0
                               ? std::min(minimum_times[1], time_taken)
                               : minimum_times[1];
        time_taken = run_a_times_b(tevv_caffeine, teov_caffeine,
                                   CoOrdinate({2}), s, true);
        std::cout << "Time taken for caffeine VVOV: " << time_taken
                  << " seconds at tile size " << s << std::endl;
        minimum_times[2] = time_taken != -1.0
                               ? std::min(minimum_times[2], time_taken)
                               : minimum_times[2];

        std::cout << "Running guanine VVOO" << std::endl;
        time_taken =
            run_a_times_b(tevv_guanine, teoo_guanine, CoOrdinate({2}), s, true);
        std::cout << "Time taken for guanine VVOO: " << time_taken
                  << " seconds at tile size " << s << std::endl;
        minimum_times[3] = time_taken != -1.0
                               ? std::min(minimum_times[3], time_taken)
                               : minimum_times[3];
        time_taken =
            run_a_times_b(teov_guanine, teov_guanine, CoOrdinate({2}), s, true);
        std::cout << "Time taken for guanine OVOV: " << time_taken
                  << " seconds at tile size " << s << std::endl;
        minimum_times[4] = time_taken != -1.0
                               ? std::min(minimum_times[4], time_taken)
                               : minimum_times[4];
        time_taken =
            run_a_times_b(tevv_guanine, teov_guanine, CoOrdinate({2}), s, true);
        std::cout << "Time taken for guanine VVOV: " << time_taken
                  << " seconds at tile size " << s << std::endl;
        minimum_times[5] = time_taken != -1.0
                               ? std::min(minimum_times[5], time_taken)
                               : minimum_times[5];
  }

  out << "caffeine VVOO"
      << "," << minimum_times[0] << std::endl;
  out << "caffeine OVOV"
      << "," << minimum_times[1] << std::endl;
  out << "caffeine VVOV"
      << "," << minimum_times[2] << std::endl;
  out << "guanine VVOO"
      << "," << minimum_times[3] << std::endl;
  out << "guanine OVOV"
      << "," << minimum_times[4] << std::endl;
  out << "guanine VVOV"
      << "," << minimum_times[5] << std::endl;
}

int main(int argc, char** argv) {
    if(argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <<path to folder containing frostt tensors>> <<path to folder containing caffeine tensors>> <<path to folder containing guanine tensors>>" << std::endl;
        return 1;
    }
    std::string frostt_dir = argv[1], caffeine_dir = argv[2], guanine_dir = argv[3];
    std::ofstream results_chem, results_frostt;
    results_chem.open("chemistry_times.csv");
    results_chem << "Tensor mode,time" << std::endl;
    std::vector<int> grid_sizes = {128, 256, 512};
    run_chemistry_experiments(grid_sizes, results_chem, caffeine_dir, guanine_dir);
    results_chem.close();
    results_frostt.open("frostt_times.csv");
    results_frostt << "Tensor mode,tile_size,acc_type,time" << std::endl;
    run_frostt_experiments(grid_sizes, results_frostt, frostt_dir);
    results_frostt.close();
    return 0;
}
