#include <nvbench/nvbench.cuh>

#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>

template <typename T>
static void in_place(nvbench::state &state, nvbench::type_list<T>)
{
  const auto num_inputs = state.get_int64("Size");
  thrust::device_vector<T> data(static_cast<size_t>(num_inputs));
  thrust::sequence(data.begin(), data.end());

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
  .add_int64_power_of_two_axis("Size", nvbench::range(16, 32, 2))
  .set_timeout(1);

NVBENCH_MAIN
