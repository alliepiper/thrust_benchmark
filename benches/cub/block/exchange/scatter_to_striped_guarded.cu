#include "block_exchange.cuh"

struct scatter_to_striped
{
  template <typename BlockExchange, typename T, int ItemsPerThread>
  __device__ int operator()(BlockExchange &block_exchange,
                            T (&thread_data)[ItemsPerThread])
  {
    int ranks[ItemsPerThread];

    const int target_thread = blockDim.x - threadIdx.x - 1;

    for (int i = 0; i < ItemsPerThread; i++)
    {
      if (i % 2)
        ranks[i] = i * blockDim.x + target_thread;
      else
        ranks[i] = -1;
    }

    block_exchange.ScatterToStripedGuarded(thread_data,
                                           thread_data,
                                           ranks);

    // Half of the items have undefined values and may exceed the limit
    return ItemsPerThread * static_cast<int>(blockDim.x) / 2;
  }
};

using op = nvbench::type_list<scatter_to_striped>;

// TODO There is a bug in ScatterToStripedGuarded with WARP_TIME_SLICE option
using supported_compute_modes =
  nvbench::enum_type_list<compute_mode::reference, compute_mode::exchange>;

NVBENCH_BENCH_TYPES(bench,
                    NVBENCH_TYPE_AXES(types,
                                      op,
                                      threads_in_block,
                                      items_per_thread,
                                      supported_compute_modes))
  .set_name("cub::BlockExchange::ScatterToStripedGuarded")
  .set_type_axes_names(block_exchange_type_axis_names())
  .add_int64_power_of_two_axis("Elements", nvbench::range(20, 22, 2));
