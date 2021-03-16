#include <cub/device/device_radix_sort.cuh>

#include <nvbench/nvbench.cuh>

#include "radix_sort_input_generator.cuh"

#include <thrust/device_vector.h>

template <typename KeyType, typename ValueType>
void sort_pairs(nvbench::state &state, nvbench::type_list<KeyType, ValueType>)
{
  // Retrieve axis parameters
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements"));

  // Enable throughput calculations and add "Size" columns to results.
  state.add_element_count(elements);
  state.add_global_memory_reads<KeyType>(elements, "KeysSize");
  state.add_global_memory_reads<ValueType>(elements, "ValuesSize");
  state.add_global_memory_writes<KeyType>(elements);
  state.add_global_memory_writes<ValueType>(elements);

  // Allocate and initialize data:
  auto input_gen = input_generator{elements, "Random"};
  thrust::device_vector<KeyType> input_keys = input_gen.generate<KeyType>();
  thrust::device_vector<KeyType> output_keys(elements, {});
  thrust::device_vector<ValueType> input_values(elements);
  thrust::device_vector<ValueType> output_values(elements, {});
  thrust::sequence(input_values.begin(), input_values.end());

  // Prepare kernel args:
  const auto *keys_in   = thrust::raw_pointer_cast(input_keys.data());
  auto *keys_out        = thrust::raw_pointer_cast(output_keys.data());
  const auto *values_in = thrust::raw_pointer_cast(input_values.data());
  auto *values_out      = thrust::raw_pointer_cast(output_values.data());
  const auto begin_bit  = int{0};
  const auto end_bit    = static_cast<int>(sizeof(KeyType) * 8);

  // Allocate temporary storage:
  std::size_t temp_size;
  cub::DeviceRadixSort::SortPairs(nullptr,
                                  temp_size,
                                  keys_in,
                                  keys_out,
                                  values_in,
                                  values_out,
                                  static_cast<int>(elements),
                                  begin_bit,
                                  end_bit);
  thrust::device_vector<nvbench::uint8_t> temp(temp_size);
  auto *temp_storage = thrust::raw_pointer_cast(temp.data());

  state.exec(nvbench::exec_tag::timer,
             [&](nvbench::launch &launch, auto &timer) {
               // Do reset:
               input_gen.reset(input_keys);
               keys_in = thrust::raw_pointer_cast(input_keys.data());

               timer.start();
               cub::DeviceRadixSort::SortPairs(temp_storage,
                                               temp_size,
                                               keys_in,
                                               keys_out,
                                               values_in,
                                               values_out,
                                               static_cast<int>(elements),
                                               begin_bit,
                                               end_bit,
                                               launch.get_stream());
               timer.stop();
             });
}
// Only spot-check a few common key types. sort_keys provides more exhaustive
// coverage.
using common_key_types =
  nvbench::type_list<nvbench::uint8_t, nvbench::int32_t, nvbench::float64_t>;
// Value types don't really matter much, they just get permuted.
// Only benchmark with a 32-bit payload.
using value_types = nvbench::type_list<nvbench::uint32_t>;
NVBENCH_BENCH_TYPES(sort_pairs,
                    NVBENCH_TYPE_AXES(common_key_types, value_types))
  .set_name("cub::DeviceRadixSort::SortPairs")
  .set_type_axes_names({"Key", "Value"})
  .add_int64_power_of_two_axis("Elements", nvbench::range(20, 30, 2));
