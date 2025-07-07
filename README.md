This repository contains the artifacts necessarty to reproduce results for the paper:

FaSTCC: Fast Sparse Tensor Contractions on CPUs

# Get the code
Clone this repository recursively so that all the dependencies are cloned correctly at the right location.
`git clone --recursive git@github.com:HPCRL/sparse_benchmark.git`

# Get the data

Download the following tensors from FROSTT:
http://frostt.io/tensors/

* NIPS
* Chicago-crime-comm
* vast-2015-mc1-5d.tns.gz
* Uber

The chemistry tensors will be packaged with the repository.

# Build the source
```
mkdir build
cd build
cmake ..
make -j16
```

# Run the experiments
Inside the build folder after running make:
```
./sc_ae_speedups <<path to frostt tensors>> <<path to caffeine tensors>> <<path to guanine tensors>>
```
This creates two CSV files in the build folder `chemistry_times.csv` `frostt_times.csv`
with the wall-time for FaSTCC kernel running with the data given.
