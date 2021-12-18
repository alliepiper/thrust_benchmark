#include "set_operations.cuh"

struct SymmetricDifference
{
  SymmetricDifference(std::size_t, std::size_t) {}

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
             OutputIterator result) const
  {
    thrust::set_symmetric_difference(exec, first1, last1, first2, last2, result);
  }
};
NVBENCH_DECLARE_TYPE_STRINGS(SymmetricDifference, "SymmetricDifference", "");

using operations = nvbench::type_list<SymmetricDifference>;

NVBENCH_BENCH_TYPES(basic, NVBENCH_TYPE_AXES(types, operations))
  .set_name("thrust::set_symmetric_difference")
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 2))
  .add_int64_axis("SetsSizeRatio", nvbench::range(1, 100, 24))
  .add_string_axis("Pattern", {"full", "empty", "random"});
