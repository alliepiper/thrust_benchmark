#include <nvbench/nvbench.h>

#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>

template <typename T>
static void BM_reduce(benchmark::State& state)
{
  thrust::device_vector<T> data(state.range(0));
  thrust::sequence(data.begin(), data.end());

  nvbench::cuda_timer timer;

  for (auto _ : state)
  {
    (void)_;
    const auto result = thrust::reduce(data.begin(), data.end());
    benchmark::DoNotOptimize(result);
  }

  state.SetItemsProcessed(data.size() * state.iterations());
  state.SetBytesProcessed(data.size() * state.iterations() * sizeof(T));
}
BENCHMARK_TEMPLATE(BM_reduce, int)->Range(1 << 12, 1ll << 29);
BENCHMARK_TEMPLATE(BM_reduce, float)->Range(1 << 12, 1ll << 29);
