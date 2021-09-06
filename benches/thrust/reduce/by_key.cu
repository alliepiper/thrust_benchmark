#include <nvbench/nvbench.cuh>

#include <thrust/device_vector.h>
#include <thrust/unique.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>

#include <tbm/range_generator.cuh>

template <typename T>
thrust::device_vector<T> gen_keys(thrust::device_vector<T> keys)
{
  std::size_t elements = keys.size();

  auto random_input =
    tbm::make_range_generator<T,
                              tbm::iterator_style::pointer,
                              tbm::data_pattern::random>(elements);

  thrust::copy(random_input.cbegin(), random_input.cend(), keys.begin());
  thrust::sort(keys.begin(), keys.end());
  return keys;
}

template <typename T>
void by_key(nvbench::state &state, nvbench::type_list<T>)
{
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements"));
  thrust::device_vector<T> in_keys(elements);
  thrust::device_vector<T> in_values(elements);

  const auto pattern = state.get_string("Pattern");
  if (pattern == "single")
  {
    // in_keys is filled with 0
  }
  else if (pattern == "random")
  {
    gen_keys(in_keys);
  }

  thrust::device_vector<T> out_values(elements);
  thrust::device_vector<T> out_keys = in_keys;

  const std::size_t unique_keys =
    thrust::distance(out_keys.begin(),
                     thrust::unique(out_keys.begin(), out_keys.end()));

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(2 * elements, "Size");
  state.add_global_memory_writes<T>(2 * unique_keys);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch &launch) {
    auto policy = thrust::device.on(launch.get_stream());

    thrust::reduce_by_key(policy,
                          in_keys.begin(),
                          in_keys.end(),
                          in_values.begin(),
                          out_keys.begin(),
                          out_values.begin());
  });
}

using types = nvbench::type_list<nvbench::uint32_t>;

NVBENCH_BENCH_TYPES(by_key, NVBENCH_TYPE_AXES(types))
  .set_name("thrust::reduce_by_key")
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 29, 2))
  .add_string_axis("Pattern", {"single", "random"});
