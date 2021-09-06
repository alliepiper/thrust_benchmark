#include <nvbench/nvbench.cuh>

#include <thrust/device_vector.h>
#include <thrust/inner_product.h>

template <typename T>
static void basic(nvbench::state &state,
                  nvbench::type_list<T>)
{
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements"));

  thrust::device_vector<T> lhs(elements);
  thrust::device_vector<T> rhs(elements);

  state.add_element_count(elements);
  state.collect_dram_throughput();
  state.collect_loads_efficiency();

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch &launch) {
    auto policy = thrust::device.on(launch.get_stream());

    thrust::inner_product(policy, lhs.begin(), lhs.end(), rhs.begin(), T{});
  });
}

using types = nvbench::type_list<nvbench::uint8_t,
                                 nvbench::uint16_t,
                                 nvbench::uint32_t,
                                 nvbench::uint64_t,
                                 nvbench::float32_t,
                                 nvbench::float64_t>;

NVBENCH_BENCH_TYPES(basic, NVBENCH_TYPE_AXES(types))
  .set_name("thrust::inner_product")
  .add_int64_power_of_two_axis("Elements", nvbench::range(20, 28, 2));
