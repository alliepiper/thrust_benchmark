#include <nvbench/nvbench.cuh>

#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>

template <typename T>
static void in_place(nvbench::state &state, nvbench::type_list<T>)
{
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements"));
  thrust::device_vector<T> data(elements);
  thrust::sequence(data.begin(), data.end());

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements, "Size");
  state.add_global_memory_writes<T>(elements);

  using namespace nvbench::exec_tag;
  state.exec(sync, [&data](nvbench::launch &launch) {
    thrust::exclusive_scan(thrust::device.on(launch.get_stream()),
                           data.cbegin(),
                           data.cend(),
                           data.begin());
  });
}
using types = nvbench::type_list<int, float, double>;
NVBENCH_BENCH_TYPES(in_place, NVBENCH_TYPE_AXES(types))
  .set_type_axes_names({"T"})
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 32, 2))
  .set_timeout(2)
  .set_skip_time(100e-6 /* us */);
