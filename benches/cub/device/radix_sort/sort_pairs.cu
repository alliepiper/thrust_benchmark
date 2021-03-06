#include <nvbench/nvbench.cuh>

#include <cub/device/device_radix_sort.cuh>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>

template <typename KeyType, typename ValueType>
void sort_pairs(nvbench::state &state, nvbench::type_list<KeyType, ValueType>)
{
  // Retrieve axis parameters
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements"));

  // Enable throughput calculations and add "Size" columns to results.
  state.add_element_count(elements);
  state.add_global_memory_reads<KeyType>(elements, "KeySize");
  state.add_global_memory_reads<ValueType>(elements, "ValueSize");
  state.add_global_memory_writes<KeyType>(elements);
  state.add_global_memory_writes<ValueType>(elements);

  // Allocate and initialize data:
  thrust::device_vector<KeyType> input_keys(elements);
  thrust::device_vector<KeyType> output_keys(elements, {});
  thrust::device_vector<ValueType> input_values(elements);
  thrust::device_vector<ValueType> output_values(elements, {});
  thrust::sequence(input_keys.begin(), input_keys.end());
  thrust::sequence(input_values.begin(), input_values.end());
  thrust::shuffle(input_keys.begin(),
                  input_keys.end(),
                  thrust::default_random_engine{});

  // Allocate temporary storage:
  std::size_t temp_size;
  cub::DeviceRadixSort::SortPairs(nullptr,
                                  temp_size,
                                  static_cast<const KeyType *>(nullptr),
                                  static_cast<KeyType *>(nullptr),
                                  static_cast<const ValueType *>(nullptr),
                                  static_cast<ValueType *>(nullptr),
                                  static_cast<int>(elements),
                                  0,
                                  sizeof(KeyType) * 8);
  thrust::device_vector<nvbench::uint8_t> temp(temp_size);

  state.exec([keys_in      = thrust::raw_pointer_cast(input_keys.data()),
              keys_out     = thrust::raw_pointer_cast(output_keys.data()),
              values_in    = thrust::raw_pointer_cast(input_values.data()),
              values_out   = thrust::raw_pointer_cast(output_values.data()),
              temp_storage = thrust::raw_pointer_cast(temp.data()),
              temp_size,
              elements](nvbench::launch &launch) {
    auto temp_size_lvalue = temp_size; // for CUB API
    cub::DeviceRadixSort::SortPairs(temp_storage,
                                    temp_size_lvalue,
                                    keys_in,
                                    keys_out,
                                    values_in,
                                    values_out,
                                    static_cast<int>(elements),
                                    0,
                                    sizeof(KeyType) * 8,
                                    launch.get_stream());
  });
}
using types = nvbench::type_list<nvbench::uint8_t,
                                 nvbench::uint16_t,
                                 nvbench::uint32_t,
                                 nvbench::uint64_t,
                                 nvbench::float32_t,
                                 nvbench::float64_t>;
NVBENCH_BENCH_TYPES(sort_pairs, NVBENCH_TYPE_AXES(types, types))
  .set_name("cub::DeviceRadixSort::SortPairs")
  .set_type_axes_names({"Key", "Value"})
  .add_int64_power_of_two_axis("Elements", nvbench::range(21, 31, 2));
