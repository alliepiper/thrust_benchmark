#include <nvbench/nvbench.cuh>

#include <cub/device/device_reduce.cuh>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>

template <typename ValueType>
void cub_device_reduce_sum(nvbench::state &state, nvbench::type_list<ValueType>)
{
  // Retrieve axis parameters
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements"));

  // Allocate and initialize data -- store output in last element:
  thrust::device_vector<ValueType> data(elements + 1);
  thrust::sequence(data.begin(), data.end());

  // Allocate temporary storage:
  std::size_t temp_size;
  cub::DeviceReduce::Sum(nullptr,
                         temp_size,
                         data.begin(),
                         data.end() - 1,
                         elements);
  thrust::device_vector<nvbench::uint8_t> temp(temp_size);

  // Enable throughput calculations and add "InputBuffer" column to results.
  state.add_element_count(elements);
  state.add_global_memory_reads<ValueType>(elements, "Size");
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
using types = nvbench::type_list<nvbench::uint8_t,
                                 nvbench::uint16_t,
                                 nvbench::uint32_t,
                                 nvbench::uint64_t,
                                 nvbench::float32_t,
                                 nvbench::float64_t>;
NVBENCH_BENCH_TYPES(cub_device_reduce_sum, NVBENCH_TYPE_AXES(types))
  .set_name("cub::DeviceReduce::Sum")
  .set_type_axes_names({"T"})
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 32, 2))
  .set_timeout(2)
  .set_skip_time(100e-6 /* us */);

NVBENCH_MAIN
