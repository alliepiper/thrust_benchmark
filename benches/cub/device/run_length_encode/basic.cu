#include <nvbench/nvbench.cuh>

#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/random.h>
#include <thrust/fill.h>

#include <tbm/range_generator.cuh>

#include <cub/device/device_run_length_encode.cuh>

template <typename T>
void basic(nvbench::state &state,
           nvbench::type_list<T>)
{
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements"));

  thrust::device_vector<T> input(elements);
  thrust::device_vector<T> unique_output(elements);
  thrust::device_vector<std::size_t> counts_output(elements);
  thrust::device_vector<std::size_t> num_runs(1);

  const auto pattern = state.get_string("Pattern");
  if (pattern == "constant")
  {
    thrust::fill(input.begin(), input.end(), T{0});
  }
  else if (pattern == "sequence")
  {
    thrust::sequence(input.begin(), input.end());
  }
  else if (pattern == "random")
  {
    auto random_input =
      tbm::make_range_generator<T,
                                tbm::iterator_style::pointer,
                                tbm::data_pattern::random>(elements);

    thrust::copy(random_input.cbegin(),
                 random_input.cend(),
                 input.begin());
    thrust::sort(input.begin(), input.end());
  }

  const T *d_input = thrust::raw_pointer_cast(input.data());
  T *d_unique_output = thrust::raw_pointer_cast(unique_output.data());
  std::size_t *d_counts_output = thrust::raw_pointer_cast(counts_output.data());
  std::size_t *d_num_runs = thrust::raw_pointer_cast(num_runs.data());

  {
    state.add_element_count(elements);
    state.add_global_memory_reads<T>(elements);

    unique_output = input;
    const auto unique_items = thrust::distance(
      unique_output.begin(),
      thrust::unique(unique_output.begin(), unique_output.end()));
    state.add_global_memory_writes<T>(unique_items);
    state.add_global_memory_writes<std::size_t>(unique_items + 1);
  }

  std::size_t temp_storage_bytes {};
  cub::DeviceRunLengthEncode::Encode(nullptr,
                                     temp_storage_bytes,
                                     d_input,
                                     d_unique_output,
                                     d_counts_output,
                                     d_num_runs,
                                     static_cast<int>(elements));

  thrust::device_vector<nvbench::uint8_t> temp_storage(temp_storage_bytes);
  nvbench::uint8_t *d_temp_storage =
    thrust::raw_pointer_cast(temp_storage.data());

  state.exec([&](nvbench::launch &launch) {
    cub::DeviceRunLengthEncode::Encode(d_temp_storage,
                                       temp_storage_bytes,
                                       d_input,
                                       d_unique_output,
                                       d_counts_output,
                                       d_num_runs,
                                       static_cast<int>(elements),
                                       launch.get_stream());
  });
}

using types = nvbench::type_list<nvbench::int8_t,
                                 nvbench::int16_t,
                                 nvbench::int32_t,
                                 nvbench::int64_t>;

NVBENCH_BENCH_TYPES(basic, NVBENCH_TYPE_AXES(types))
  .set_name("cub::DeviceRunLengthEncode::Encode")
  .add_int64_power_of_two_axis("Elements", nvbench::range(20, 30, 2))
  .add_string_axis("Pattern", {"const", "sequence", "random"});