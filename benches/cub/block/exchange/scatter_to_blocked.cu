#include "block_exchange.cuh"

struct scatter_to_blocked
{
  template <typename BlockExchange, typename T, int ItemsPerThread>
  __device__ void operator()(BlockExchange &block_exchange,
                             T (&thread_data)[ItemsPerThread])
  {
    int ranks[ItemsPerThread];

    const int target_thread = blockDim.x - threadIdx.x - 1;

    for (int i = 0; i < ItemsPerThread; i++)
    {
      ranks[i] = target_thread * ItemsPerThread + i;
    }

    block_exchange.ScatterToBlocked(thread_data, thread_data, ranks);
  }
};

using op = nvbench::type_list<scatter_to_blocked>;

NVBENCH_BENCH_TYPES(bench,
                    NVBENCH_TYPE_AXES(types,
                                      op,
                                      threads_in_block,
                                      items_per_thread,
                                      compute_modes))
  .set_name("cub::BlockExchange::ScatterToBlocked")
  .set_type_axes_names(block_exchange_type_axis_names())
  .add_int64_power_of_two_axis("Blocks", nvbench::range(20, 22, 2));
