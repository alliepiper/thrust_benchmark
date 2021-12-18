#include "search.cuh"

struct BinaryBound
{
  template <typename DerivedPolicy,
            typename ForwardIterator,
            typename InputIterator,
            typename OutputIterator>
  __host__ __device__ void
  operator()(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
             ForwardIterator first,
             ForwardIterator last,
             InputIterator values_first,
             InputIterator values_last,
             OutputIterator output)
  {
    thrust::binary_search(exec, first, last, values_first, values_last, output);
  }
};

using operations = nvbench::type_list<BinaryBound>;

NVBENCH_BENCH_TYPES(search, NVBENCH_TYPE_AXES(types, operations))
  .set_name("thrust::binary_search")
  .add_int64_axis("NeedlesRatio", nvbench::range(1, 100, 24))
  .add_int64_power_of_two_axis("Elements", nvbench::range(20, 26, 2))
  .add_string_axis("Pattern", {"sequence", "uniform", "random"});
