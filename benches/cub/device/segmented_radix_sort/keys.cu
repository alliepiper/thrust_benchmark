#include "segments_generator.cuh"
#include "type_lists.cuh"

#include <nvbench/nvbench.cuh>

#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>

#include <cub/device/device_segmented_radix_sort.cuh>

#include <type_traits>

template <typename T, sort_direction SortDirection>
void basic(nvbench::state &state,
           nvbench::type_list<T, nvbench::enum_type<SortDirection>>)
{
  const int elements = static_cast<int>(state.get_int64("Elements"));

  thrust::device_vector<T> input(elements);
  thrust::device_vector<T> output(elements);

  const auto pattern = state.get_string("Pattern");
  const auto offsets = gen_offsets(pattern, elements);

  const auto bits = state.get_string("Bits");

  // cub::DeviceSegmentedRadixSort reads data multiple times. Limiting the
  // number of bits is a way of having accurate throughput estimation.
  const int first_bit = 0;
  const int last_bit  = bits == "all" ? sizeof(T) * 8 : 4;

  const int num_segments = static_cast<int>(offsets.size() - 1);

  const T *d_input                   = thrust::raw_pointer_cast(input.data());
  T *d_output                        = thrust::raw_pointer_cast(output.data());
  const nvbench::uint32_t *d_offsets = thrust::raw_pointer_cast(offsets.data());

  std::size_t temp_storage_bytes{};
  if constexpr (SortDirection == sort_direction::ascending)
  {
    cub::DeviceSegmentedRadixSort::SortKeys(nullptr,
                                            temp_storage_bytes,
                                            d_input,
                                            d_output,
                                            elements,
                                            num_segments,
                                            d_offsets,
                                            d_offsets + 1,
                                            first_bit,
                                            last_bit);
  }
  else
  {
    cub::DeviceSegmentedRadixSort::SortKeysDescending(nullptr,
                                                      temp_storage_bytes,
                                                      d_input,
                                                      d_output,
                                                      elements,
                                                      num_segments,
                                                      d_offsets,
                                                      d_offsets + 1,
                                                      first_bit,
                                                      last_bit);
  }

  thrust::device_vector<nvbench::uint8_t> temp_storage(temp_storage_bytes);
  nvbench::uint8_t *d_temp_storage =
    thrust::raw_pointer_cast(temp_storage.data());

  state.add_element_count(elements);
  state.add_global_memory_reads<nvbench::uint32_t>(num_segments, "Segments");
  state.add_global_memory_reads<T>(elements);
  state.add_global_memory_writes<T>(elements);

  state.exec([&](nvbench::launch &launch) {
    if constexpr (SortDirection == sort_direction::ascending)
    {
      cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage,
                                              temp_storage_bytes,
                                              d_input,
                                              d_output,
                                              elements,
                                              num_segments,
                                              d_offsets,
                                              d_offsets + 1,
                                              first_bit,
                                              last_bit,
                                              launch.get_stream());
    }
    else
    {
      cub::DeviceSegmentedRadixSort::SortKeysDescending(d_temp_storage,
                                                        temp_storage_bytes,
                                                        d_input,
                                                        d_output,
                                                        elements,
                                                        num_segments,
                                                        d_offsets,
                                                        d_offsets + 1,
                                                        first_bit,
                                                        last_bit,
                                                        launch.get_stream());
    }
  });
}

NVBENCH_BENCH_TYPES(basic, NVBENCH_TYPE_AXES(types, directions))
  .set_name("cub::DeviceSegmentedRadixSort::SortKeys")
  .add_int64_power_of_two_axis("Elements", nvbench::range(20, 30, 2))
  .add_string_axis("Bits", {"few", "all"})
  .add_string_axis("Pattern", {"small", "large", "random"});
