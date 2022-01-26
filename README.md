# Build Instruction

To build Thrust's benchmark suite, the following recipe can be used:

```
# Check out thrust_benchmark and thrust:
git clone --recursive https://github.com/NVIDIA/thrust.git
git clone --recursive https://github.com/allisonvacanti/thrust_benchmark

# Configure build
mkdir thrust_benchmark_build
cd thrust_benchmark_build
cmake ../thrust_benchmark \
      -DThrust_DIR=../thrust/thrust/cmake

# Compile
make

# Lock gpu clocks, enable CUPTI permissions, etc
# For example, on Linux and Volta+:
# sudo bin/nvbench-ctl --pm 1 --lgc base

# Run benchmarks
ctest -L "bench"

# Reset locked GPU clocks. Example:
# sudo bin/nvbench-ctl --pm 0 --lgc reset

# View results
ls results/
```

This will build all benchmarks executables, which are named
`thrust_benchmark_build/bin/bench.*`.
