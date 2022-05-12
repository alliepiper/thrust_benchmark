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
      ranks[i] = i * blockDim.x + target_thread;
    }

    block_exchange.ScatterToStriped(thread_data, thread_data, ranks);

    return 0; // All items have defined values
  }
};

using op = nvbench::type_list<scatter_to_striped>;

NVBENCH_BENCH_TYPES(bench,
                    NVBENCH_TYPE_AXES(types,
                                      op,
                                      threads_in_block,
                                      items_per_thread,
                                      compute_modes))
  .set_name("cub::BlockExchange::ScatterToStriped")
  .set_type_axes_names(block_exchange_type_axis_names())
  .add_int64_power_of_two_axis("Elements", nvbench::range(20, 22, 2));
