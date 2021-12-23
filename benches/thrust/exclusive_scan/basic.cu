#include <nvbench/nvbench.cuh>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>

template <typename T>
void basic(nvbench::state &state, nvbench::type_list<T>)
{
  const auto elements = state.get_int64("Elements");
  thrust::device_vector<T> input(elements);
  thrust::device_vector<T> output(elements);
  thrust::sequence(input.begin(), input.end());

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements, "Size");
  state.add_global_memory_writes<T>(elements);

  state.exec(nvbench::exec_tag::sync, // thrust algos sync internally
             [&input, &output](nvbench::launch &launch) {
               thrust::exclusive_scan(thrust::device.on(launch.get_stream()),
                                      input.cbegin(),
                                      input.cend(),
                                      output.begin());
             });
}
using types = nvbench::type_list<nvbench::int8_t,
                                 nvbench::int16_t,
                                 nvbench::int32_t,
                                 nvbench::int64_t,
                                 nvbench::float32_t,
                                 nvbench::float64_t>;
NVBENCH_BENCH_TYPES(basic, NVBENCH_TYPE_AXES(types))
  .set_name("thrust::exclusive_scan")
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 30, 2));
