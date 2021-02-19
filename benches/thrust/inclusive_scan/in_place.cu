#include <nvbench/nvbench.cuh>

#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>

template <typename T>
void in_place(nvbench::state &state, nvbench::type_list<T>)
{
  const auto num_inputs =
    static_cast<std::size_t>(state.get_int64("Elements"));
  thrust::device_vector<T> data(num_inputs);
  thrust::sequence(data.begin(), data.end());

  state.add_element_count(num_inputs);
  state.add_global_memory_reads<T>(num_inputs, "Size");
  state.add_global_memory_writes<T>(num_inputs);

  state.exec(nvbench::exec_tag::sync, // thrust algos synchronize
             [&data](nvbench::launch &launch) {
               thrust::inclusive_scan(thrust::device.on(launch.get_stream()),
                                      data.cbegin(),
                                      data.cend(),
                                      data.begin());
             });
}
using types = nvbench::type_list<nvbench::uint8_t,
                                 nvbench::uint16_t,
                                 nvbench::uint32_t,
                                 nvbench::uint64_t,
                                 nvbench::float32_t,
                                 nvbench::float64_t>;
NVBENCH_BENCH_TYPES(in_place, NVBENCH_TYPE_AXES(types))
  .set_name("thrust::inclusive_scan (in-place)")
  .set_type_axes_names({"T"})
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 32, 2))
  // Speed things up:
  .set_skip_time(50e-6 /* 50 us */)
  .set_timeout(2);
