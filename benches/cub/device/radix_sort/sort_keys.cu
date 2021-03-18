#include "sort_keys_bench.cuh"

#include <nvbench/nvbench.cuh>

// Benchmark all radix_sortable types with a large variety of inputs:
NVBENCH_BENCH_TYPES(sort_keys,
                    NVBENCH_TYPE_AXES(all_key_types,
                                      random_input,
                                      ascending_sort))
  .set_name("cub::DeviceRadixSort::SortKeys - Overview")
  .set_type_axes_names(sort_keys_type_axis_names())
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 30, 2));

// Benchmark constant values:
NVBENCH_BENCH_TYPES(sort_keys,
                    NVBENCH_TYPE_AXES(unique_size_key_types,
                                      constant_input,
                                      ascending_sort))
  .set_name("cub::DeviceRadixSort::SortKeys - Constant Values")
  .set_type_axes_names(sort_keys_type_axis_names())
  .add_int64_power_of_two_axis("Elements", nvbench::range(20, 30, 2));

// Benchmark only sorting half of the bits:
NVBENCH_BENCH_TYPES(sort_keys,
                    NVBENCH_TYPE_AXES(unique_size_key_types,
                                      random_input,
                                      ascending_sort))
  .set_name("cub::DeviceRadixSort::SortKeys - Half Word")
  .set_type_axes_names(sort_keys_type_axis_names())
  .add_int64_power_of_two_axis("Elements", nvbench::range(20, 30, 2))
  .add_string_axis("Bits", {"Half"});
