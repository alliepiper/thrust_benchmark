#include "unique.cuh"

struct Unique
{
  Unique(std::size_t) {}

  template <typename DerivedPolicy,
            typename InputIterator,
            typename OutputIterator>
  void
  operator()(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
             InputIterator first,
             InputIterator last,
             OutputIterator result)
  {
    thrust::unique_copy(exec, first, last, result);
  }

  std::size_t extra_bytes_to_read()
  {
    return 0;
  }

  template <typename T>
  std::size_t extra_bytes_to_write(std::size_t)
  {
    return 0;
  }
};

using operations = nvbench::type_list<Unique>;

NVBENCH_BENCH_TYPES(basic, NVBENCH_TYPE_AXES(types, operations))
  .set_name("thrust::unique_copy")
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 2))
  .add_string_axis("Pattern", {"unique", "const", "random"});
