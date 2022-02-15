# Build Instruction

To build and run Thrust's benchmark suite, the following recipe can be used:

```
# Check out thrust_benchmark and thrust:
git clone --recursive https://github.com/NVIDIA/thrust.git
git clone --recursive https://github.com/allisonvacanti/thrust_benchmark

# Configure build
mkdir thrust_benchmark_build
cd thrust_benchmark_build
# See https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html
# for info on setting CMAKE_CUDA_ARCHITECTURES. tl;dr, this is a
# semicolon-separated list of SM architectures to generate PTX/SASS for.
# Example: `-DCMAKE_CUDA_ARCHITECTURES="60-virtual;61-real;70"` will produce:
# - PTX for sm_60
# - SASS for sm_61
# - both PTX and SAS for sm_70
cmake ../thrust_benchmark \
      -DCMAKE_CUDA_ARCHITECTURES=<Target arches here!> \
      -DThrust_DIR=../thrust/thrust/cmake

# Compile -- builds NVBench and the bin/bench.* executables
make

# Lock gpu clocks, enable CUPTI permissions, etc
# For example, on Linux and Volta+, persistence mode and clocks can be
# configured with NVBench directly:
#
# sudo bin/nvbench-ctl --pm 1 --lgc base
#
# CUPTI permissions are more complicated. See instructions here:
#
# https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters#SolnAdminTag

# Run all benchmarks.
# By default, JSON, CSV, and Markdown outputs are written to result/*
ctest -L "bench"

# Reset locked GPU clocks, etc. Example:
# sudo bin/nvbench-ctl --pm 0 --lgc reset

# View results
ls results/
```

To compare with previous runs, use the python scipt at `thrust_benchmark/nvbench/scripts/nvbench-compare.py`:

```
# Single json file:
nvbench-compare.py /path/to/baseline/benchmark.json /path/to/new/benchmark.json

# Entire directory:
nvbench-compare.py /path/to/baseline/json-file-dir/ /path/to/new/json-file-dir/
```
