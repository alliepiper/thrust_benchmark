#include <nvbench/nvbench.cuh>

#include "reduce_bench.cuh"

// Sweep though various input iterators and patterns using cub::Max and
// a small number of value_types.
NVBENCH_BENCH_TYPES(reduce,
                    NVBENCH_TYPE_AXES(common_value_types,
                                      cub_max_op,
                                      all_input_iter_styles,
                                      all_input_data_patterns))
  .set_name("cub::DeviceReduce::Reduce - Overview")
  .set_type_axes_names(reduce_type_axis_names())
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 30, 2));
