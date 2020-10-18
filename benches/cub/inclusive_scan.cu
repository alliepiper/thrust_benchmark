#include <nvbench/nvbench.h>

#include <thrust/device_vector.h>
#include <thrust/sequence.h>

// Why is this in detail?
#include <thrust/detail/raw_pointer_cast.h>

#include <cub/device/device_scan.cuh>

#include <cuda/std/cstdint>

template <typename T>
static void BM_inclusive_scan(benchmark::State &state)
{
  using byte = cuda::std::uint8_t;

  thrust::device_vector<T> input(state.range(0));
  thrust::device_vector<T> output(state.range(0));
  thrust::sequence(input.begin(), input.end());

  auto const *inPtr = thrust::raw_pointer_cast(input.data());
  auto *outPtr      = thrust::raw_pointer_cast(output.data());

  for (auto _ : state)
  {
    (void)_;
    size_t tmp_size;
    cub::DeviceScan::InclusiveSum(nullptr,
                                  tmp_size,
                                  inPtr,
                                  outPtr,
                                  static_cast<int>(input.size()));
    thrust::device_vector<byte> tmp(tmp_size);
    cub::DeviceScan::InclusiveSum(thrust::raw_pointer_cast(tmp.data()),
                                  tmp_size,
                                  input.cbegin(),
                                  output.begin(),
                                  static_cast<int>(input.size()));
    // implicit sync in tmp's destructor
  }

  state.SetItemsProcessed(input.size() * state.iterations());
  state.SetBytesProcessed(input.size() * state.iterations() * sizeof(T));
}
BENCHMARK_TEMPLATE(BM_inclusive_scan, int)->Range(1 << 12, 1ll << 28);
BENCHMARK_TEMPLATE(BM_inclusive_scan, float)->Range(1 << 12, 1ll << 28);

template <typename T>
static void BM_inclusive_scan_reuse_tmp(benchmark::State &state)
{
  using byte = cuda::std::uint8_t;

  thrust::device_vector<T> input(state.range(0));
  thrust::device_vector<T> output(state.range(0));
  thrust::sequence(input.begin(), input.end());

  auto const *inPtr = thrust::raw_pointer_cast(input.data());
  auto *outPtr      = thrust::raw_pointer_cast(output.data());

  nvbench::cuda_timer timer;

  size_t tmp_size;
  cub::DeviceScan::InclusiveSum(nullptr,
                                tmp_size,
                                inPtr,
                                outPtr,
                                static_cast<int>(input.size()));
  thrust::device_vector<byte> tmp(tmp_size);

  for (auto _ : state)
  {
    (void)_;

    timer.start();
    cub::DeviceScan::InclusiveSum(thrust::raw_pointer_cast(tmp.data()),
                                  tmp_size,
                                  inPtr,
                                  outPtr,
                                  static_cast<int>(input.size()));
    timer.stop();
    // Implicit sync in seconds_elapsed()
    state.SetIterationTime(timer.seconds_elapsed());
  }

  state.SetItemsProcessed(input.size() * state.iterations());
  state.SetBytesProcessed(input.size() * state.iterations() * sizeof(T));
}
BENCHMARK_TEMPLATE(BM_inclusive_scan_reuse_tmp, int)
  ->UseManualTime()
  ->Range(1 << 12, 1ll << 28);
BENCHMARK_TEMPLATE(BM_inclusive_scan_reuse_tmp, float)
  ->UseManualTime()
  ->Range(1 << 12, 1ll << 28);
