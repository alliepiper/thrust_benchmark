#include <nvbench/nvbench.h>

#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>

template <typename T>
static void TBM_in_place(benchmark::State &state)
{
  thrust::device_vector<T> data(state.range(0));
  thrust::sequence(data.begin(), data.end());

  for (auto _ : state)
  {
    (void)_;
    thrust::exclusive_scan(data.cbegin(), data.cend(), data.begin());
  }

  state.SetItemsProcessed(data.size() * state.iterations());
  state.SetBytesProcessed(data.size() * state.iterations() * sizeof(T));
}
BENCHMARK_TEMPLATE(TBM_in_place, int)->Range(1 << 12, 1ll << 28);
BENCHMARK_TEMPLATE(TBM_in_place, float)->Range(1 << 12, 1ll << 28);
