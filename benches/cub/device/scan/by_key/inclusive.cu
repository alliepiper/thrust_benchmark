#include <nvbench/nvbench.cuh>

#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

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

  T *d_keys       = thrust::raw_pointer_cast(keys.data());
  T *d_in_values  = thrust::raw_pointer_cast(in_values.data());
  T *d_out_values = thrust::raw_pointer_cast(out_values.data());

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(2 * elements); // read keys and values
  state.add_global_memory_writes<T>(elements);    // write values

  std::size_t tmp_size{};
  cub::DeviceScan::InclusiveSumByKey(nullptr,
                                     tmp_size,
                                     d_keys,
                                     d_in_values,
                                     d_out_values,
                                     static_cast<int>(elements),
                                     cub::Equality{});
  thrust::device_vector<nvbench::uint8_t> tmp(tmp_size);

  std::uint8_t *d_tmp = thrust::raw_pointer_cast(tmp.data());

  state.exec([&](nvbench::launch &launch) {
    const std::size_t temp_size = tmp.size();

    cub::DeviceScan::InclusiveSumByKey(d_tmp,
                                       tmp_size,
                                       d_keys,
                                       d_in_values,
                                       d_out_values,
                                       static_cast<int>(elements),
                                       cub::Equality{},
                                       launch.get_stream());
  });
}
using types = nvbench::type_list<nvbench::int8_t,
                                 nvbench::int16_t,
                                 nvbench::int32_t,
                                 nvbench::int64_t>;
NVBENCH_BENCH_TYPES(by_key, NVBENCH_TYPE_AXES(types))
  .set_name("cub::DeviceScan::InclusiveSumByKey")
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 2))
  .add_string_axis("Pattern", {"single", "random"});
