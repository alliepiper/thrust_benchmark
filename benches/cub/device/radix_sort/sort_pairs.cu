#include "sort_pairs_bench.cuh"

#include <nvbench/nvbench.cuh>

// Only spot-check a few common key types. sort_keys provides more exhaustive
// coverage.
NVBENCH_BENCH_TYPES(sort_pairs,
                    NVBENCH_TYPE_AXES(common_key_types,
                                      value_types,
                                      random_input,
                                      ascending_sort))
  .set_name("cub::DeviceRadixSort::SortPairs")
  .set_type_axes_names(sort_pairs_type_axis_names())
  .add_int64_power_of_two_axis("Elements", nvbench::range(20, 30, 2));
