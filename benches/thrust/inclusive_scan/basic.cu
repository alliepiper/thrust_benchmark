#include <nvbench/nvbench.h>

#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>

template <typename T>
static void TBM_basic(benchmark::State &state)
{
  thrust::device_vector<T> input(state.range(0));
  thrust::device_vector<T> output(state.range(0));
  thrust::sequence(input.begin(), input.end());

  for (auto _ : state)
  {
    (void)_;
    thrust::inclusive_scan(input.cbegin(), input.cend(), output.begin());
  }

  state.SetItemsProcessed(input.size() * state.iterations());
  state.SetBytesProcessed(input.size() * state.iterations() * sizeof(T));
}
BENCHMARK_TEMPLATE(TBM_basic, int)->Range(1 << 12, 1ll << 28);
BENCHMARK_TEMPLATE(TBM_basic, float)->Range(1 << 12, 1ll << 28);
