#include <cub/device/device_radix_sort.cuh>

#include <nvbench/nvbench.cuh>

#include "radix_sort_input_generator.cuh"

#include <thrust/device_vector.h>

template <typename KeyType>
void sort_keys(nvbench::state &state, nvbench::type_list<KeyType>)
{
  // Retrieve axis parameters
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements"));
  const auto distribution = state.get_string_or_default("Input", "Random");
  const auto bits         = state.get_string_or_default("Bits", "All");

  // Enable throughput calculations and add "Size" column
  // to results.
  state.add_element_count(elements);
  state.add_global_memory_reads<KeyType>(elements, "Size");
  state.add_global_memory_writes<KeyType>(elements);

  // Allocate and initialize data:
  auto input_gen = input_generator{elements, distribution};
  thrust::device_vector<KeyType> input = input_gen.generate<KeyType>();
  thrust::device_vector<KeyType> output(elements, {});

  const auto key_bits     = static_cast<int>(sizeof(KeyType) * 8);
  const auto bits_divisor = bits == "Half" ? 2 : 1;

  // Prepare kernel args:
  const auto *keys_in  = thrust::raw_pointer_cast(input.data());
  auto *keys_out       = thrust::raw_pointer_cast(output.data());
  const auto begin_bit = int{0};
  const auto end_bit   = key_bits / bits_divisor;

  // Allocate temporary storage:
  std::size_t temp_size;
  cub::DeviceRadixSort::SortKeys(nullptr,
                                 temp_size,
                                 keys_in,
                                 keys_out,
                                 static_cast<int>(elements),
                                 begin_bit,
                                 end_bit);
  thrust::device_vector<nvbench::uint8_t> temp(temp_size);
  auto *temp_storage = thrust::raw_pointer_cast(temp.data());

  // Call a different overload of `exec` depending on whether or not we need
  // a timer to cut out a reset of the input.
  // FIXME:
  // The input distribution could be moved to a `enum_type_list` and this
  // branch could be constexpr; it'd avoid unused instantiations for some types.
  if (!input_gen.needs_reset())
  {
    state.exec([&](nvbench::launch &launch) {
      cub::DeviceRadixSort::SortKeys(temp_storage,
                                     temp_size,
                                     keys_in,
                                     keys_out,
                                     static_cast<int>(elements),
                                     begin_bit,
                                     end_bit,
                                     launch.get_stream());
    });
  }
  else // needs timer/reset:
  {
    state.exec(nvbench::exec_tag::timer,
               [&](nvbench::launch &launch, auto &timer) {
                 // Do reset:
                 input_gen.reset(input);
                 keys_in = thrust::raw_pointer_cast(input.data());

                 timer.start();
                 cub::DeviceRadixSort::SortKeys(temp_storage,
                                                temp_size,
                                                keys_in,
                                                keys_out,
                                                static_cast<int>(elements),
                                                begin_bit,
                                                end_bit,
                                                launch.get_stream());
                 timer.stop();
               });
  }
}

// All radix sortable types
using all_types = nvbench::type_list<bool,
                                     nvbench::uint8_t,
                                     nvbench::uint16_t,
                                     nvbench::uint32_t,
                                     nvbench::uint64_t,
                                     nvbench::int8_t,
                                     nvbench::int16_t,
                                     nvbench::int32_t,
                                     nvbench::int64_t,
                                     // FIXME: thrust/random doesn't handle half
                                     // __half,
                                     nvbench::float32_t,
                                     nvbench::float64_t>;

// Uniquely sized radix sortable types
using unique_size_types = nvbench::type_list<nvbench::uint8_t,
                                             nvbench::uint16_t,
                                             nvbench::uint32_t,
                                             nvbench::uint64_t>;

// Benchmark all radix_sortable types with a large variety of inputs:
NVBENCH_BENCH_TYPES(sort_keys, NVBENCH_TYPE_AXES(all_types))
  .set_name("cub::DeviceRadixSort::SortKeys - Overview")
  .add_int64_power_of_two_axis("Elements", nvbench::range(20, 30, 2))
  .add_string_axis("Input", {"Random"});

// Benchmark constant values:
NVBENCH_BENCH_TYPES(sort_keys, NVBENCH_TYPE_AXES(unique_size_types))
  .set_name("cub::DeviceRadixSort::SortKeys - Constant Values")
  .add_int64_power_of_two_axis("Elements", nvbench::range(20, 30, 2))
  .add_string_axis("Input", {"Constant"});

// Benchmark only sorting half of the bits:
NVBENCH_BENCH_TYPES(sort_keys, NVBENCH_TYPE_AXES(unique_size_types))
  .set_name("cub::DeviceRadixSort::SortKeys - Half Word")
  .add_int64_power_of_two_axis("Elements", nvbench::range(20, 30, 2))
  .add_string_axis("Input", {"Random"})
  .add_string_axis("Bits", {"Half"});
