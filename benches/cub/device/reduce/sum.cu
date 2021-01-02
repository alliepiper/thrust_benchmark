#include <nvbench/nvbench.cuh>

#include <cub/device/device_reduce.cuh>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>

template <typename ValueType>
void cub_device_reduce_sum(nvbench::state &state, nvbench::type_list<ValueType>)
{
  // Retrieve axis parameters
  const auto num_inputs = state.get_int64("NumInputs");

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

  nvbench::exec(state,
                [&data, &temp, &num_inputs](nvbench::launch& launch) {
                  auto an_lvalue = temp.size();
                  cub::DeviceReduce::Sum(thrust::raw_pointer_cast(temp.data()),
                                         an_lvalue,
                                         data.begin(),
                                         data.end() - 1,
                                         data.size() - 1,
                                         launch.get_stream());
                });
}
NVBENCH_CREATE_TEMPLATE(
  cub_device_reduce_sum,
  NVBENCH_TYPE_AXES(nvbench::type_list<int, float, double>))
  .set_name("cub::DeviceReduce::Sum")
  .set_type_axes_names({"ValueType"})
  .add_int64_power_of_two_axis("NumInputs", {10, 20});

NVBENCH_MAIN
