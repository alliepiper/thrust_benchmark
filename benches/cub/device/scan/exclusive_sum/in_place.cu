#include <nvbench/nvbench.h>

#include <thrust/device_vector.h>
#include <thrust/sequence.h>

// Why is this in detail?
#include <thrust/detail/raw_pointer_cast.h>

#include <cub/device/device_scan.cuh>

#include <cuda/std/cstdint>

template <typename T>
static void TBM_alloc_sync(benchmark::State &state)
{
  using byte = cuda::std::uint8_t;

  thrust::device_vector<T> data(state.range(0));
  thrust::sequence(data.begin(), data.end());

  auto const *inPtr = thrust::raw_pointer_cast(data.data());
  auto *outPtr      = thrust::raw_pointer_cast(data.data());

  for (auto _ : state)
  {
    (void)_;
    size_t tmp_size;
    cub::DeviceScan::ExclusiveSum(nullptr,
                                  tmp_size,
                                  inPtr,
                                  outPtr,
                                  static_cast<int>(data.size()));
    thrust::device_vector<byte> tmp(tmp_size);
    cub::DeviceScan::ExclusiveSum(thrust::raw_pointer_cast(tmp.data()),
                                  tmp_size,
                                  inPtr,
                                  outPtr,
                                  static_cast<int>(data.size()));
    // implicit sync in tmp's destructor
  }

  state.SetItemsProcessed(data.size() * state.iterations());
  state.SetBytesProcessed(data.size() * state.iterations() * sizeof(T));
}
BENCHMARK_TEMPLATE(TBM_alloc_sync, int)->Range(1 << 12, 1ll << 28);
BENCHMARK_TEMPLATE(TBM_alloc_sync, float)->Range(1 << 12, 1ll << 28);

template <typename T>
static void TBM_noalloc_nosync(benchmark::State &state)
{
  using byte = cuda::std::uint8_t;

  thrust::device_vector<T> data(state.range(0));
  thrust::sequence(data.begin(), data.end());

  auto const *inPtr = thrust::raw_pointer_cast(data.data());
  auto *outPtr      = thrust::raw_pointer_cast(data.data());

  nvbench::cuda_timer timer;

  size_t tmp_size;
  cub::DeviceScan::ExclusiveSum(nullptr,
                                tmp_size,
                                inPtr,
                                outPtr,
                                static_cast<int>(data.size()));
  thrust::device_vector<byte> tmp(tmp_size);

  for (auto _ : state)
  {
    (void)_;

    timer.start();
    cub::DeviceScan::ExclusiveSum(thrust::raw_pointer_cast(tmp.data()),
                                  tmp_size,
                                  inPtr,
                                  outPtr,
                                  static_cast<int>(data.size()));
    timer.stop();
    // Implicit sync in seconds_elapsed()
    state.SetIterationTime(timer.seconds_elapsed());
  }

  state.SetItemsProcessed(data.size() * state.iterations());
  state.SetBytesProcessed(data.size() * state.iterations() * sizeof(T));
}
BENCHMARK_TEMPLATE(TBM_noalloc_nosync, int)
  ->UseManualTime()
  ->Range(1 << 12, 1ll << 28);
BENCHMARK_TEMPLATE(TBM_noalloc_nosync, float)
  ->UseManualTime()
  ->Range(1 << 12, 1ll << 28);
