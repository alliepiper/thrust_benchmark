#include <nvbench/nvbench.cuh>

#include <thrust/device_vector.h>
#include <thrust/copy.h>

template <typename T>
static void basic(nvbench::state &state,
                  nvbench::type_list<T>)
{
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements"));

  thrust::device_vector<T> input(elements);
  thrust::device_vector<T> output(elements);

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements);
  state.add_global_memory_writes<T>(elements);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch &launch) {
    auto policy = thrust::device.on(launch.get_stream());

    thrust::copy(policy,
                 input.cbegin(),
                 input.cend(),
                 output.begin());
  });
}

using types = nvbench::type_list<nvbench::uint8_t,
                                 nvbench::uint16_t,
                                 nvbench::uint32_t,
                                 nvbench::uint64_t>;

NVBENCH_BENCH_TYPES(basic, NVBENCH_TYPE_AXES(types))
  .set_name("thrust::copy")
  .add_int64_power_of_two_axis("Elements", nvbench::range(20, 28, 2));
