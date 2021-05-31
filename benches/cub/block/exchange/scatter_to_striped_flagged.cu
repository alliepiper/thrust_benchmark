#include "block_exchange.cuh"

struct scatter_to_striped
{
  template <typename BlockExchange, typename T, int ItemsPerThread>
  __device__ void operator()(BlockExchange &block_exchange,
                             T (&thread_data)[ItemsPerThread])
  {
    int ranks[ItemsPerThread];
    bool is_valid[ItemsPerThread];

    const int target_thread = blockDim.x - threadIdx.x - 1;

    for (int i = 0; i < ItemsPerThread; i++)
    {
      ranks[i] = i * blockDim.x + target_thread;
      is_valid[i] = i % 2;
    }

    block_exchange.ScatterToStripedFlagged(thread_data,
                                           thread_data,
                                           ranks,
                                           is_valid);
  }
};

using op = nvbench::type_list<scatter_to_striped>;

// TODO There is a bug in ScatterToStripedFlagged with WARP_TIME_SLICE option
using supported_compute_modes =
  nvbench::enum_type_list<compute_mode::reference, compute_mode::exchange>;

NVBENCH_BENCH_TYPES(bench,
                    NVBENCH_TYPE_AXES(types,
                                      op,
                                      threads_in_block,
                                      items_per_thread,
                                      supported_compute_modes))
  .set_name("cub::BlockExchange::ScatterToStripedFlagged")
  .set_type_axes_names(block_exchange_type_axis_names())
  .add_int64_power_of_two_axis("Blocks", nvbench::range(20, 22, 2));
