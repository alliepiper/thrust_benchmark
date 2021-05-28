#include <nvbench/nvbench.cuh>

#include "reduce_bench.cuh"

template <typename DataType, unsigned int OperationsLeft>
struct helper;

template <typename DataType>
struct helper<DataType, 1>
{
  static __device__ DataType compute(DataType lhs, DataType rhs)
  {
    return lhs + rhs;
  }
};

template <typename DataType, unsigned int OperationsLeft>
struct helper
{
  static __device__ DataType compute(DataType lhs, DataType rhs)
  {
    return helper<DataType, OperationsLeft - 1>::compute(sqrt(lhs * lhs),
                                                         sqrt(rhs * rhs));
  }
};

template <unsigned int OperationsCount>
struct complex_op
{
  template <typename DataType>
  __device__ DataType operator()(DataType lhs, DataType rhs)
  {
    return helper<DataType, OperationsCount>::compute(lhs, rhs);
  }
};

using cub_custom_op = nvbench::type_list<complex_op<128>>;
using fp_value_types = nvbench::type_list<nvbench::float32_t>;

// Sweep though various input iterators and patterns using cub::Max and
// a small number of value_types.
NVBENCH_BENCH_TYPES(reduce,
                    NVBENCH_TYPE_AXES(fp_value_types,
                                      cub_custom_op,
                                      pointer_iter_style,
                                      sequence_data_pattern))
  .set_name("cub::DeviceReduce::Reduce - Complex operator")
  .set_type_axes_names(reduce_type_axis_names())
  .add_int64_power_of_two_axis("Elements", nvbench::range(22, 28, 2));
