#include "block_exchange.cuh"

struct blocked_to_warp_striped
{
  template <typename BlockExchange, typename T, int ItemsPerThread>
  __device__ void operator()(BlockExchange &block_exchange,
                             T (&thread_data)[ItemsPerThread])
  {
    block_exchange.BlockedToWarpStriped(thread_data, thread_data);
  }
};

using op = nvbench::type_list<blocked_to_warp_striped>;

NVBENCH_BENCH_TYPES(bench,
                    NVBENCH_TYPE_AXES(types,
                                      op,
                                      threads_in_block,
                                      items_per_thread,
                                      compute_modes))
  .set_name("cub::BlockExchange::BlockedToWarpStriped")
  .set_type_axes_names(block_exchange_type_axis_names())
  .add_int64_power_of_two_axis("Elements", nvbench::range(20, 22, 2));
