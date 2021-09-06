#include "unique.cuh"

#include <type_traits>

struct UniqueByKey
{
  thrust::device_vector<std::uint8_t> in_values;
  thrust::device_vector<std::uint8_t> out_values;

  UniqueByKey(std::size_t bytes)
      : in_values(bytes)
      , out_values(bytes)
  {}

  std::size_t extra_bytes_to_read()
  {
    return in_values.size();
  }

  template <typename T>
  std::size_t extra_bytes_to_write(std::size_t unique_elements)
  {
    return unique_elements * sizeof(T);
  }

  template <typename DerivedPolicy,
            typename InputIterator,
            typename OutputIterator>
  void
  operator()(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
             InputIterator first,
             InputIterator last,
             OutputIterator result)
  {
    using KeyT = typename std::iterator_traits<InputIterator>::value_type;
    using ValueT = KeyT;

    auto d_in_values =
      reinterpret_cast<ValueT *>(thrust::raw_pointer_cast(in_values.data()));
    auto d_out_values =
      reinterpret_cast<ValueT *>(thrust::raw_pointer_cast(out_values.data()));

    thrust::unique_by_key_copy(exec,
                               first,
                               last,
                               d_in_values,
                               result,
                               d_out_values);
  }
};

using operations = nvbench::type_list<UniqueByKey>;

NVBENCH_BENCH_TYPES(basic, NVBENCH_TYPE_AXES(types, operations))
  .set_name("thrust::unique_by_key_copy")
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 2))
  .add_string_axis("Pattern", {"unique", "const", "random"});
