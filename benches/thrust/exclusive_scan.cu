#include <nvbench/nvbench.h>

#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>

template <typename T>
static void BM_exclusive_scan(benchmark::State& state)
{
  thrust::device_vector<T> input(state.range(0));
  thrust::device_vector<T> output(state.range(0));
  thrust::sequence(input.begin(), input.end());

  nvbench::cuda_timer timer;

  for (auto _ : state)
  {
    (void)_;
    thrust::exclusive_scan(input.cbegin(), input.cend(), output.begin());
  }

  state.SetItemsProcessed(input.size() * state.iterations());
  state.SetBytesProcessed(input.size() * state.iterations() * sizeof(T));
}
BENCHMARK_TEMPLATE(BM_exclusive_scan, int)->Range(1 << 12, 1ll << 28);
BENCHMARK_TEMPLATE(BM_exclusive_scan, float)->Range(1 << 12, 1ll << 28);
