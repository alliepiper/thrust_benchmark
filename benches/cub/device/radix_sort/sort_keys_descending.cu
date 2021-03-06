#include <nvbench/nvbench.cuh>

#include <cub/device/device_radix_sort.cuh>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>

template <typename KeyType>
void sort_keys_descending(nvbench::state &state, nvbench::type_list<KeyType>)
{
  // Retrieve axis parameters
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements"));

  // Enable throughput calculations and add "Size" column to results.
  state.add_element_count(elements);
  state.add_global_memory_reads<KeyType>(elements, "Size");
  state.add_global_memory_writes<KeyType>(elements);

  // Allocate and initialize data:
  thrust::device_vector<KeyType> input(elements);
  thrust::device_vector<KeyType> output(elements, {});
  thrust::sequence(input.begin(), input.end());
  thrust::shuffle(input.begin(), input.end(), thrust::default_random_engine{});

  // Allocate temporary storage:
  std::size_t temp_size;
  cub::DeviceRadixSort::SortKeysDescending(nullptr,
                                           temp_size,
                                           static_cast<const KeyType *>(
                                             nullptr),
                                           static_cast<KeyType *>(nullptr),
                                           static_cast<int>(elements),
                                           0,
                                           sizeof(KeyType) * 8);
  thrust::device_vector<nvbench::uint8_t> temp(temp_size);

  state.exec([keys_in      = thrust::raw_pointer_cast(input.data()),
              keys_out     = thrust::raw_pointer_cast(output.data()),
              temp_storage = thrust::raw_pointer_cast(temp.data()),
              temp_size,
              elements](nvbench::launch &launch) {
    auto temp_size_lvalue = temp_size; // for CUB API
    cub::DeviceRadixSort::SortKeysDescending(temp_storage,
                                             temp_size_lvalue,
                                             keys_in,
                                             keys_out,
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
NVBENCH_BENCH_TYPES(sort_keys_descending, NVBENCH_TYPE_AXES(types))
  .set_name("cub::DeviceRadixSort::SortKeysDescending")
  .add_int64_power_of_two_axis("Elements", nvbench::range(21, 31, 2));
