#include "set_operations.cuh"

struct Union
{
  Union(std::size_t, std::size_t) {}

  template <typename DerivedPolicy,
            typename InputIterator1,
            typename InputIterator2,
            typename OutputIterator>
  __host__ __device__ OutputIterator
  operator()(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
             InputIterator1 first1,
             InputIterator1 last1,
             InputIterator2 first2,
             InputIterator2 last2,
             OutputIterator result) const
  {
    return thrust::set_union(exec, first1, last1, first2, last2, result);
  }
};
NVBENCH_DECLARE_TYPE_STRINGS(Union, "Union", "");

using operations = nvbench::type_list<Union>;

NVBENCH_BENCH_TYPES(basic, NVBENCH_TYPE_AXES(types, operations))
  .set_name("thrust::set_union")
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 2))
  .add_int64_axis("SetsSizeRatio", nvbench::range(1, 100, 24))
  .add_string_axis("Pattern", {"full", "empty", "random"});
