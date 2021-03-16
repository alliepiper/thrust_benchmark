#include <cub/device/device_radix_sort.cuh>

#include <nvbench/nvbench.cuh>

#include "radix_sort_input_generator.cuh"

#include <thrust/device_vector.h>

template <typename KeyType>
void sort_keys_descending(nvbench::state &state, nvbench::type_list<KeyType>)
{
  // Retrieve axis parameters
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements"));

  // Enable throughput calculations and add "Size" column
  // to results.
  state.add_element_count(elements);
  state.add_global_memory_reads<KeyType>(elements, "Size");
  state.add_global_memory_writes<KeyType>(elements);

  // Allocate and initialize data:
  auto input_gen                       = input_generator{elements, "Random"};
  thrust::device_vector<KeyType> input = input_gen.generate<KeyType>();
  thrust::device_vector<KeyType> output(elements, {});

  // Prepare kernel args:
  const auto *keys_in  = thrust::raw_pointer_cast(input.data());
  auto *keys_out       = thrust::raw_pointer_cast(output.data());
  const auto begin_bit = int{0};
  const auto end_bit   = static_cast<int>(sizeof(KeyType) * 8);

  // Allocate temporary storage:
  std::size_t temp_size;
  cub::DeviceRadixSort::SortKeysDescending(nullptr,
                                           temp_size,
                                           keys_in,
                                           keys_out,
                                           static_cast<int>(elements),
                                           begin_bit,
                                           end_bit);
  thrust::device_vector<nvbench::uint8_t> temp(temp_size);
  auto *temp_storage = thrust::raw_pointer_cast(temp.data());

  state.exec(nvbench::exec_tag::timer,
             [&](nvbench::launch &launch, auto &timer) {
               // Do reset:
               input_gen.reset(input);
               keys_in = thrust::raw_pointer_cast(input.data());

               timer.start();
               cub::DeviceRadixSort::SortKeysDescending(temp_storage,
                                                        temp_size,
                                                        keys_in,
                                                        keys_out,
                                                        static_cast<int>(
                                                          elements),
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
NVBENCH_BENCH_TYPES(sort_keys_descending, NVBENCH_TYPE_AXES(common_key_types))
  .set_name("cub::DeviceRadixSort::SortKeysDescending")
  .add_int64_power_of_two_axis("Elements", nvbench::range(20, 30, 2));
