#include "set_operations.cuh"

#include <thrust/device_vector.h>
#include <type_traits>


struct DifferenceByKey
{
  thrust::device_vector<std::uint8_t> A_values;
  thrust::device_vector<std::uint8_t> values_result;

  DifferenceByKey(std::size_t A_bytes,
                  std::size_t)
    : A_values(A_bytes)
    , values_result(A_bytes)
  {

  }

  template <typename DerivedPolicy,
            typename InputIterator1,
            typename InputIterator2,
            typename OutputIterator>
  __host__ __device__ void
  operator()(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
             InputIterator1 first1,
             InputIterator1 last1,
             InputIterator2 first2,
             InputIterator2 last2,
             OutputIterator keys_result)
  {
    using KeyT = typename std::iterator_traits<InputIterator1>::value_type;
    using ValueT = KeyT;

    auto d_A_values = reinterpret_cast<const ValueT *>(
      thrust::raw_pointer_cast(A_values.data()));
    auto d_values_result = reinterpret_cast<ValueT *>(
      thrust::raw_pointer_cast(values_result.data()));

    // B values aren't needed (issues #1493)
    thrust::set_difference_by_key(exec,
                                  first1,
                                  last1,
                                  first2,
                                  last2,
                                  d_A_values,
                                  thrust::make_counting_iterator(ValueT{}),
                                  keys_result,
                                  d_values_result);
  }
};
NVBENCH_DECLARE_TYPE_STRINGS(DifferenceByKey, "DifferenceByKey", "");

using operations = nvbench::type_list<DifferenceByKey>;

NVBENCH_BENCH_TYPES(basic, NVBENCH_TYPE_AXES(types, operations))
  .set_name("thrust::set_difference_by_key")
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 2))
  .add_int64_axis("SetsSizeRatio", nvbench::range(1, 100, 24))
  .add_string_axis("Pattern", {"full", "empty", "random"});
