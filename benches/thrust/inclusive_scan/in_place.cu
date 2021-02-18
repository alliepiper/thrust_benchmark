#include <nvbench/nvbench.cuh>

#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>

template <typename T>
void in_place(nvbench::state &state, nvbench::type_list<T>)
{
  const auto num_inputs =
    static_cast<std::size_t>(state.get_int64("NumInputs"));
  thrust::device_vector<T> data(num_inputs);
  thrust::sequence(data.begin(), data.end());

  state.add_element_count(num_inputs);
  state.add_global_memory_reads<T>(num_inputs, "InputSize");
  state.add_global_memory_writes<T>(num_inputs);

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
