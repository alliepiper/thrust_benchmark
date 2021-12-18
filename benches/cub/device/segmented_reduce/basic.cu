#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/random.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include <tbm/range_generator.cuh>

#include <cub/device/device_segmented_reduce.cuh>

thrust::device_vector<nvbench::uint32_t> gen_offsets(int segment_size,
                                                     int elements)
{
  const int num_segments = (elements + segment_size - 1) / segment_size;
  thrust::device_vector<nvbench::uint32_t> offsets(num_segments + 1);
  thrust::fill(offsets.begin(),
               offsets.end(),
               static_cast<nvbench::uint32_t>(segment_size));
  thrust::exclusive_scan(offsets.begin(), offsets.end(), offsets.begin());
  offsets[num_segments] = elements;
  return offsets;
}

template <typename T>
struct SegmentSizeLimiter
{
  T out_max;

  SegmentSizeLimiter(T out_max)
      : out_max(out_max)
  {}

  __device__ T operator()(T in)
  {
    T guarded_value = in % out_max;
    return guarded_value < 1 ? 1 : guarded_value;
  }
};

thrust::device_vector<nvbench::uint32_t>
gen_random(nvbench::uint32_t max_segment_size, int elements)
{
  thrust::device_vector<nvbench::uint32_t> offsets(elements);

  auto random_input =
    tbm::make_range_generator<nvbench::uint32_t,
                              tbm::iterator_style::pointer,
                              tbm::data_pattern::random>(elements);

  thrust::transform(random_input.cbegin(),
                    random_input.cend(),
                    offsets.begin(),
                    SegmentSizeLimiter<nvbench::uint32_t>{max_segment_size});

  thrust::exclusive_scan(offsets.begin(), offsets.end(), offsets.begin());
  const auto num_segments = thrust::distance(
    offsets.begin(),
    thrust::lower_bound(offsets.begin(),
                        thrust::is_sorted_until(offsets.begin(), offsets.end()),
                        static_cast<nvbench::uint32_t>(elements)));

  offsets.resize(num_segments + 1);
  offsets[num_segments] = elements;

  return offsets;
}

thrust::device_vector<nvbench::uint32_t> gen_offsets(const std::string &pattern,
                                                     int elements)
{
  if (pattern == "small")
  {
    return gen_offsets(4, elements);
  }
  else if (pattern == "large")
  {
    return gen_offsets(256 * 1024, elements);
  }

  return gen_random(16 * 1024, elements);
}

template <typename T>
void basic(nvbench::state &state, nvbench::type_list<T>)
{
  const int elements = static_cast<int>(state.get_int64("Elements"));

  thrust::device_vector<T> input(elements);
  thrust::device_vector<T> output(elements);

  const auto pattern = state.get_string("Pattern");
  const auto offsets = gen_offsets(pattern, elements);

  const int num_segments = static_cast<int>(offsets.size() - 1);

  const T *d_input                   = thrust::raw_pointer_cast(input.data());
  T *d_output                        = thrust::raw_pointer_cast(output.data());
  const nvbench::uint32_t *d_offsets = thrust::raw_pointer_cast(offsets.data());

  std::size_t temp_storage_bytes{};
  cub::DeviceSegmentedReduce::Sum(nullptr,
                                  temp_storage_bytes,
                                  d_input,
                                  d_output,
                                  num_segments,
                                  d_offsets,
                                  d_offsets + 1);

  thrust::device_vector<nvbench::uint8_t> temp_storage(temp_storage_bytes);
  nvbench::uint8_t *d_temp_storage =
    thrust::raw_pointer_cast(temp_storage.data());

  state.add_element_count(elements);
  state.add_global_memory_reads<nvbench::uint32_t>(num_segments, "Segments");
  state.add_global_memory_reads<T>(elements);
  state.add_global_memory_writes<T>(num_segments);

  state.exec([&](nvbench::launch &launch) {
    cub::DeviceSegmentedReduce::Sum(d_temp_storage,
                                    temp_storage_bytes,
                                    d_input,
                                    d_output,
                                    num_segments,
                                    d_offsets,
                                    d_offsets + 1,
                                    launch.get_stream());
  });
}

using types = nvbench::type_list<nvbench::uint8_t,
                                 nvbench::uint16_t,
                                 nvbench::uint32_t,
                                 nvbench::uint64_t>;

NVBENCH_BENCH_TYPES(basic, NVBENCH_TYPE_AXES(types))
  .set_name("cub::DeviceSegmentedReduce::Sum")
  .add_int64_power_of_two_axis("Elements", nvbench::range(20, 28, 2))
  .add_string_axis("Pattern", {"small", "large", "random"});
