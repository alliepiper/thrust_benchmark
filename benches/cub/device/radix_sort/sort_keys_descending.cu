#include "sort_keys_bench.cuh"

#include <nvbench/nvbench.cuh>

// Only spot-check a few common key types. sort_keys provides more exhaustive
// coverage.
NVBENCH_BENCH_TYPES(sort_keys,
                    NVBENCH_TYPE_AXES(common_key_types,
                                      random_input,
                                      descending_sort))
  .set_name("cub::DeviceRadixSort::SortKeysDescending")
  .set_type_axes_names(sort_keys_type_axis_names())
  .add_int64_power_of_two_axis("Elements", nvbench::range(20, 30, 2));
