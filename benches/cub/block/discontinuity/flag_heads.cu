#include "block_discontinuity.cuh"

struct heads
{
  template <typename BlockDiscontinuity,
            typename T,
            int ItemsPerThread>
  __device__ void operator()(BlockDiscontinuity &block_discontinuity,
                             bool (&flags)[ItemsPerThread],
                             T (&thread_data)[ItemsPerThread])
  {
    block_discontinuity.FlagHeads(flags, thread_data, cub::Inequality());
  }
};
NVBENCH_DECLARE_TYPE_STRINGS(heads, "heads", "");

using op = nvbench::type_list<heads>;

NVBENCH_BENCH_TYPES(bench,
                    NVBENCH_TYPE_AXES(types,
                                      op,
                                      threads_in_block,
                                      items_per_thread))
  .set_name("cub::BlockDiscontinuity::FlagHeads")
  .set_type_axes_names(type_axis_names())
  .add_string_axis("Mode", {"Reference", "Compute"});
