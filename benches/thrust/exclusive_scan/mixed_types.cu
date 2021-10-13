#include <nvbench/nvbench.cuh>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>

using value_types = nvbench::type_list<nvbench::int32_t, nvbench::float32_t>;

template <typename InputType, typename OutputType, typename InitialValueType>
void mixed_types(nvbench::state &state,
                 nvbench::type_list<InputType, OutputType, InitialValueType>)
{
  const auto size = static_cast<std::size_t>(state.get_int64("Size"));

  thrust::device_vector<InputType> input(size);
  thrust::device_vector<OutputType> output(size);

  thrust::sequence(input.begin(), input.end());

  state.add_element_count(size);
  state.add_global_memory_reads<InputType>(size);
  state.add_global_memory_writes<OutputType>(size);

  state.exec(nvbench::exec_tag::sync,
             [&input, &output](nvbench::launch &launch) {
               thrust::exclusive_scan(thrust::device.on(launch.get_stream()),
                                      input.cbegin(),
                                      input.cend(),
                                      output.begin());
             });
}
NVBENCH_BENCH_TYPES(mixed_types,
                    NVBENCH_TYPE_AXES(value_types, value_types, value_types))
  .set_name("thrust::exclusive_scan (mixed types)")
  .set_type_axes_names({"In", "Out", "Init"})
  .add_int64_power_of_two_axis("Size", nvbench::range(20, 30, 4))
  .set_timeout(2)
  .set_skip_time(100e-6 /* us */);
