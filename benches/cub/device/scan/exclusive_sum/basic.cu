#include <nvbench/nvbench.cuh>

#include <thrust/device_vector.h>
#include <thrust/sequence.h>

// Why is this in detail?
#include <thrust/detail/raw_pointer_cast.h>

#include <cub/device/device_scan.cuh>

template <typename T>
static void basic(nvbench::state &state, nvbench::type_list<T>)
{
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements"));

  thrust::device_vector<T> input(elements);
  thrust::device_vector<T> output(elements);

  thrust::sequence(input.begin(), input.end());

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements, "Size");
  state.add_global_memory_writes<T>(elements);

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
using types = nvbench::type_list<nvbench::uint8_t,
                                 nvbench::uint16_t,
                                 nvbench::uint32_t,
                                 nvbench::uint64_t,
                                 nvbench::float32_t,
                                 nvbench::float64_t>;
NVBENCH_BENCH_TYPES(basic, NVBENCH_TYPE_AXES(types))
  .set_name("cub::DeviceScan::ExclusiveSum (copy)")
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 32, 2))
  .set_timeout(2)
  .set_skip_time(100e-6 /* us */);
