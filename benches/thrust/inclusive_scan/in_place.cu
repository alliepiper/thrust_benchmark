#include <nvbench/nvbench.cuh>

#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>

template <typename T>
void in_place(nvbench::state &state, nvbench::type_list<T>)
{
  const auto num_inputs = state.get_int64("NumInputs");
  thrust::device_vector<T> data(num_inputs);
  thrust::sequence(data.begin(), data.end());

  const auto num_bytes = num_inputs * sizeof(T);
  state.set_global_bytes_accessed_per_launch(2 * num_bytes);
  state.set_items_processed_per_launch(num_inputs);

  auto &buffer_size_col = state.add_summary("Input Size");
  buffer_size_col.set_string("hint", "bytes");
  buffer_size_col.set_int64("value", num_bytes);

  state.exec(nvbench::exec_tag::sync, // thrust algos synchronize
             [&data](nvbench::launch &launch) {
               thrust::inclusive_scan(thrust::device.on(launch.get_stream()),
                                      data.cbegin(),
                                      data.cend(),
                                      data.begin());
             });
}
NVBENCH_BENCH_TYPES(in_place, NVBENCH_TYPE_AXES(nvbench::type_list<double>))
  .set_name("thrust::inclusive_scan (in-place)")
  .set_type_axes_names({"T"})
  .add_int64_power_of_two_axis("NumInputs", nvbench::range(22, 25, 1))
  .set_timeout(1);

NVBENCH_MAIN
