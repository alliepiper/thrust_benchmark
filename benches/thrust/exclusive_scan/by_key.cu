#include <nvbench/nvbench.cuh>

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/scan.h>

#include <tbm/range_generator.cuh>

template <typename T>
void gen_keys(thrust::device_vector<T> &keys)
{
  std::size_t elements = keys.size();

  auto random_input =
    tbm::make_range_generator<T,
                              tbm::iterator_style::pointer,
                              tbm::data_pattern::random>(elements);

  thrust::copy(random_input.cbegin(), random_input.cend(), keys.begin());
  thrust::sort(keys.begin(), keys.end());
}

template <typename T>
void by_key(nvbench::state &state, nvbench::type_list<T>)
{
  const auto elements = state.get_int64("Elements");
  thrust::device_vector<T> keys(elements);
  thrust::device_vector<T> in_values(elements);
  thrust::device_vector<T> out_values(elements);

  const auto pattern = state.get_string("Pattern");
  if (pattern == "single")
  {
    // keys is filled with 0
  }
  else if (pattern == "random")
  {
    gen_keys(keys);
  }

  state.add_element_count(elements);
  state.collect_cupti_metrics();

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch &launch) {
    auto policy = thrust::device.on(launch.get_stream());
    thrust::exclusive_scan_by_key(policy,
                                  keys.cbegin(),
                                  keys.cend(),
                                  in_values.cbegin(),
                                  out_values.begin(),
                                  T{42});
  });
}
using types = nvbench::type_list<nvbench::int8_t,
                                 nvbench::int16_t,
                                 nvbench::int32_t,
                                 nvbench::int64_t>;
NVBENCH_BENCH_TYPES(by_key, NVBENCH_TYPE_AXES(types))
  .set_name("thrust::exclusive_scan_by_key")
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 30, 2))
  .add_string_axis("Pattern", {"single", "random"});
