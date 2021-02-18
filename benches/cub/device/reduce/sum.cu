#include <nvbench/nvbench.cuh>

#include <cub/device/device_reduce.cuh>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>

template <typename ValueType>
void cub_device_reduce_sum(nvbench::state &state, nvbench::type_list<ValueType>)
{
  // Retrieve axis parameters
  const auto num_inputs =
    static_cast<std::size_t>(state.get_int64("NumInputs"));

  // Allocate and initialize data -- store output in last element:
  thrust::device_vector<ValueType> data(num_inputs + 1);
  thrust::sequence(data.begin(), data.end());

  // Allocate temporary storage:
  std::size_t temp_size;
  cub::DeviceReduce::Sum(nullptr,
                         temp_size,
                         data.begin(),
                         data.end() - 1,
                         num_inputs);
  thrust::device_vector<nvbench::uint8_t> temp(temp_size);

  // Enable throughput calculations and add "InputBuffer" column to results.
  state.add_element_count(num_inputs);
  state.add_global_memory_reads<ValueType>(num_inputs, "InputBuffer");
  state.add_global_memory_writes<ValueType>(1);

  state.exec([&data, &temp, &temp_size](nvbench::launch &launch) {
    cub::DeviceReduce::Sum(thrust::raw_pointer_cast(temp.data()),
                           temp_size,
                           data.begin(),
                           data.end() - 1,
                           data.size() - 1,
                           launch.get_stream());
  });
}
NVBENCH_BENCH_TYPES(cub_device_reduce_sum,
                    NVBENCH_TYPE_AXES(nvbench::type_list<int, float, double>))
  .set_name("cub::DeviceReduce::Sum")
  .set_type_axes_names({"ValueType"})
  .add_int64_power_of_two_axis("NumInputs", nvbench::range(20, 28, 4));

NVBENCH_MAIN
