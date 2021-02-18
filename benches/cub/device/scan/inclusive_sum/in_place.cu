#include <nvbench/nvbench.cuh>

#include <thrust/device_vector.h>
#include <thrust/sequence.h>

// Why is this in detail?
#include <thrust/detail/raw_pointer_cast.h>

#include <cub/device/device_scan.cuh>

template <typename T>
static void in_place(nvbench::state &state, nvbench::type_list<T>)
{
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements"));

  thrust::device_vector<T> data(elements);
  thrust::sequence(data.begin(), data.end());

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements, "Size");
  state.add_global_memory_writes<T>(elements);

  size_t tmp_size;
  cub::DeviceScan::InclusiveSum(nullptr,
                                tmp_size,
                                data.cbegin(),
                                data.begin(),
                                static_cast<int>(data.size()));
  thrust::device_vector<nvbench::uint8_t> tmp(tmp_size);

  state.exec([&data, &tmp](nvbench::launch &launch) {
    std::size_t temp_size = tmp.size(); // need an lvalue
    cub::DeviceScan::InclusiveSum(thrust::raw_pointer_cast(tmp.data()),
                                  temp_size,
                                  data.cbegin(),
                                  data.begin(),
                                  static_cast<int>(data.size()),
                                  launch.get_stream());
  });
}
using types = nvbench::type_list<nvbench::uint8_t,
                                 nvbench::uint16_t,
                                 nvbench::uint32_t,
                                 nvbench::uint64_t,
                                 nvbench::float32_t,
                                 nvbench::float64_t>;
NVBENCH_BENCH_TYPES(in_place, NVBENCH_TYPE_AXES(types))
  .set_name("cub::DeviceScan::InclusiveSum (in_place)")
  .set_type_axes_names({"T"})
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 32, 2))
  .set_timeout(2)
  .set_skip_time(100e-6 /* us */);

NVBENCH_MAIN
