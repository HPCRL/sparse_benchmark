cmake_minimum_required(VERSION 3.16)

project(sparse_benchmark)

#random link

set(CMAKE_THREAD_LIBS_INIT "-lpthread")
set(CMAKE_HAVE_THREADS_LIBRARY 1)
set(CMAKE_USE_WIN32_THREADS_INIT 0)
set(CMAKE_USE_PTHREADS_INIT 1)
set(THREADS_PREFER_PTHREAD_FLAG ON)

# Abseil requires C++14
set(CMAKE_CXX_STANDARD 20)
#set(CMAKE_CXX_COMPILER "/usr/bin/clang++")
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++ -lc++abi")

# Process Abseil's CMake build system
add_subdirectory(./sparse_opcount/abseil-cpp)
add_subdirectory(./sparse_opcount/unordered_dense)
include_directories(~/hopscotch-map ~/taskflow ~/taskflow/taskflow/ ${CMAKE_CURRENT_SOURCE_DIR}/sparse_opcount /usr/include/c++/11 /usr/include/x86_64-linux-gnu/c++/11)

add_executable(benchmark benchmark.cc)
#add_executable(tests tests.cc)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mtune=native -g")

target_include_directories(benchmark PUBLIC ~/taskflow ~/taskflow/taskflow/ ~/hopscotch-map/include ${CMAKE_CURRENT_SOURCE_DIR}/sparse_opcount)

# Declare dependency on the absl::strings library
target_link_libraries(benchmark unordered_dense::unordered_dense absl::container_memory absl::hash_container_defaults absl::raw_hash_set absl::algorithm_container absl::core_headers absl::memory absl::type_traits)
