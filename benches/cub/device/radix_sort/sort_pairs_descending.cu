#include "sort_pairs_bench.cuh"

#include <nvbench/nvbench.cuh>

NVBENCH_BENCH_TYPES(sort_pairs,
                    NVBENCH_TYPE_AXES(common_key_types,
                                      value_types,
                                      random_input,
                                      descending_sort))
  .set_name("cub::DeviceRadixSort::SortPairsDescending")
  .set_type_axes_names(sort_pairs_type_axis_names())
  .add_int64_power_of_two_axis("Elements", nvbench::range(20, 30, 2));
