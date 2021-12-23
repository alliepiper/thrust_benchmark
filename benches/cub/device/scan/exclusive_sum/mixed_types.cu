#include <nvbench/nvbench.cuh>

#include <thrust/device_vector.h>
#include <thrust/sequence.h>

#include <thrust/detail/raw_pointer_cast.h>

#include <cub/device/device_scan.cuh>

template <typename InputType, typename OutputType>
void mixed_types(nvbench::state &state,
                 nvbench::type_list<InputType, OutputType>)
{
  const auto size = static_cast<std::size_t>(state.get_int64("Elements"));

  thrust::device_vector<InputType> input(size);
  thrust::device_vector<OutputType> output(size);

  thrust::sequence(input.begin(), input.end());

  state.add_global_memory_reads<InputType>(size, "InputSize");
  state.add_global_memory_writes<OutputType>(size, "OutputSize");
  state.add_element_count(size);

  size_t tmp_size;
  cub::DeviceScan::ExclusiveSum(nullptr,
                                tmp_size,
                                input.cbegin(),
                                output.begin(),
                                static_cast<int>(input.size()));
  thrust::device_vector<nvbench::uint8_t> tmp(tmp_size);

  state.exec([&input, &output, &tmp](nvbench::launch &launch) {
    std::size_t temp_size = tmp.size(); // need an lvalue
    cub::DeviceScan::ExclusiveSum(thrust::raw_pointer_cast(tmp.data()),
                                  temp_size,
                                  input.cbegin(),
                                  output.begin(),
                                  static_cast<int>(input.size()),
                                  launch.get_stream());
  });
}
using value_types = nvbench::type_list<nvbench::int32_t, nvbench::float32_t>;
template <typename T>
void mixed_types(nvbench::state &state, nvbench::type_list<T, T>)
{
  state.skip("Types are not mixed.");
}
NVBENCH_BENCH_TYPES(mixed_types, NVBENCH_TYPE_AXES(value_types, value_types))
  .set_name("cub::DeviceScan::ExclusiveSum (mixed types)")
  .set_type_axes_names({"In", "Out"})
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 30, 2));
