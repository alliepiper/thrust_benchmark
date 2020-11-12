#include <nvbench/nvbench.h>

#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>

// Why is this in detail?
#include <thrust/detail/raw_pointer_cast.h>

#include <cub/device/device_radix_sort.cuh>

#include <cuda/std/cstdint>

template <typename T>
static void TBM_alloc_sync(benchmark::State &state)
{
  using byte = cuda::std::uint8_t;

  thrust::device_vector<T> input(state.range(0));
  thrust::device_vector<T> output(state.range(0));
  thrust::sequence(input.begin(), input.end());
  thrust::shuffle(input.begin(), input.end(), thrust::default_random_engine{0});

  auto const *inPtr = thrust::raw_pointer_cast(input.data());
  auto *outPtr      = thrust::raw_pointer_cast(output.data());

  for (auto _ : state)
  {
    (void)_;
    size_t tmp_size;
    cub::DeviceRadixSort::SortKeys(nullptr,
				   tmp_size,
				   inPtr,
				   outPtr,
				   static_cast<int>(input.size()));
    thrust::device_vector<byte> tmp(tmp_size);
    cub::DeviceRadixSort::SortKeys(thrust::raw_pointer_cast(tmp.data()),
				   tmp_size,
				   inPtr,
				   outPtr,
				   static_cast<int>(input.size()));
    // implicit sync in tmp's destructor
  }

  state.SetItemsProcessed(input.size() * state.iterations());
  state.SetBytesProcessed(input.size() * state.iterations() * sizeof(T));
}
BENCHMARK_TEMPLATE(TBM_alloc_sync, cuda::std::int16_t)->Range(1 << 12, 1ll << 28);
//BENCHMARK_TEMPLATE(TBM_alloc_sync, int)->Range(1 << 12, 1ll << 28);
//BENCHMARK_TEMPLATE(TBM_alloc_sync, float)->Range(1 << 12, 1ll << 28);

template <typename T>
static void TBM_noalloc_nosync(benchmark::State &state)
{
  using byte = cuda::std::uint8_t;

  thrust::device_vector<T> input(state.range(0));
  thrust::device_vector<T> output(state.range(0));
  thrust::sequence(input.begin(), input.end());
  thrust::shuffle(input.begin(), input.end(), thrust::default_random_engine{0});

  auto const *inPtr = thrust::raw_pointer_cast(input.data());
  auto *outPtr      = thrust::raw_pointer_cast(output.data());

  nvbench::cuda_timer timer;

  size_t tmp_size;
  cub::DeviceRadixSort::SortKeys(nullptr,
				 tmp_size,
				 inPtr,
				 outPtr,
				 static_cast<int>(input.size()));
  thrust::device_vector<byte> tmp(tmp_size);

  for (auto _ : state)
  {
    (void)_;

    timer.start();
    cub::DeviceRadixSort::SortKeys(thrust::raw_pointer_cast(tmp.data()),
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
BENCHMARK_TEMPLATE(TBM_noalloc_nosync, cuda::std::int16_t)
  ->UseManualTime()
  ->Range(1 << 12, 1ll << 28);
// BENCHMARK_TEMPLATE(TBM_noalloc_nosync, int)
//   ->UseManualTime()
//   ->Range(1 << 12, 1ll << 28);
// BENCHMARK_TEMPLATE(TBM_noalloc_nosync, float)
//   ->UseManualTime()
//   ->Range(1 << 12, 1ll << 28);
