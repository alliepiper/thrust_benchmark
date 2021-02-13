#include <nvbench/nvbench.cuh>

#include <thrust/device_vector.h>
#include <thrust/sequence.h>

#include <thrust/detail/raw_pointer_cast.h>

#include <cub/device/device_scan.cuh>

using value_types = nvbench::type_list<nvbench::int32_t, nvbench::float32_t>;

template <typename InputType, typename OutputType>
void mixed_types(nvbench::state &state,
                 nvbench::type_list<InputType, OutputType>)
{
  const auto size = state.get_int64("Size");

  thrust::device_vector<InputType> input(size);
  thrust::device_vector<OutputType> output(size);

  thrust::sequence(input.begin(), input.end());

  const auto input_bytes  = size * sizeof(InputType);
  const auto output_bytes = size * sizeof(OutputType);

  auto &input_size_summary = state.add_summary("InputSize");
  input_size_summary.set_string("hint", "bytes");
  input_size_summary.set_int64("value",
                               static_cast<nvbench::int64_t>(input_bytes));

  state.set_global_bytes_accessed_per_launch(input_bytes + output_bytes);
  state.set_items_processed_per_launch(size);

  size_t tmp_size;
  cub::DeviceScan::ExclusiveSum(nullptr,
                                tmp_size,
                                input.cbegin(),
                                output.begin(),
                                static_cast<int>(input.size()));
  thrust::device_vector<nvbench::uint8_t> tmp(tmp_size);

  nvbench::exec(state, [&input, &output, &tmp](nvbench::launch &launch) {
    std::size_t temp_size = tmp.size(); // need an lvalue
    cub::DeviceScan::ExclusiveSum(thrust::raw_pointer_cast(tmp.data()),
                                  temp_size,
                                  input.cbegin(),
                                  output.begin(),
                                  static_cast<int>(input.size()),
                                  launch.get_stream());
  });
}
NVBENCH_CREATE_TEMPLATE(mixed_types,
                        NVBENCH_TYPE_AXES(value_types, value_types))
  .set_name("cub::DeviceScan::ExclusiveSum (mixed types)")
  .set_type_axes_names({"In", "Out"})
  .add_int64_power_of_two_axis("Size", nvbench::range(24, 28, 2));

NVBENCH_MAIN;
