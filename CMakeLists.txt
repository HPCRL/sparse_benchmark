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
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -fsanitize=address -g")

# Process Abseil's CMake build system
add_subdirectory(./fastcc/abseil-cpp)
add_subdirectory(./fastcc/unordered_dense)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/hopscotch-map ${CMAKE_CURRENT_SOURCE_DIR}/taskflow ${CMAKE_CURRENT_SOURCE_DIR}/taskflow/taskflow/ ${CMAKE_CURRENT_SOURCE_DIR}/fastcc)


#/usr/include/c++/11 /usr/include/x86_64-linux-gnu/c++/11

add_executable(sc_ae_speedups sc_ae_speedups.cc)
#add_executable(tests tests.cc)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mtune=native -g -fopenmp -lstdc++")

target_include_directories(sc_ae_speedups PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/taskflow ${CMAKE_CURRENT_SOURCE_DIR}/taskflow/taskflow/ ${CMAKE_CURRENT_SOURCE_DIR}/hopscotch-map/include ${CMAKE_CURRENT_SOURCE_DIR}/fastcc ${CMAKE_CURRENT_SOURCE_DIR}/fastcc/emhash)

# Declare dependency on the absl::strings library
target_link_libraries(sc_ae_speedups unordered_dense::unordered_dense absl::container_memory absl::hash_container_defaults absl::raw_hash_set absl::algorithm_container absl::core_headers absl::memory absl::type_traits)

