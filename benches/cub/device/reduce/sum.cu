#include <nvbench/nvbench.cuh>

#include "reduce_bench.cuh"

// Compare `cub::Sum` to `custom_sum` for all value_types:
NVBENCH_BENCH_TYPES(reduce,
                    NVBENCH_TYPE_AXES(all_value_types,
                                      sum_ops,
                                      pointer_iter_style,
                                      sequence_data_pattern))
  .set_name("cub::DeviceReduce::Reduce - cub::Sum vs CustomSum")
  .set_type_axes_names(reduce_type_axis_names())
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 30, 2));
