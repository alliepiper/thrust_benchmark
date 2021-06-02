#include <nvbench/nvbench.cuh>

#include <thrust/device_vector.h>
#include <thrust/sequence.h>

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

template <typename T, int OperationsCount>
static void
complex_op_bench(nvbench::state &state,
                 nvbench::type_list<T, nvbench::enum_type<OperationsCount>>)
{
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements"));

  thrust::device_vector<T> input(elements);
  thrust::device_vector<T> output(elements);

  thrust::sequence(input.begin(), input.end());

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements, "Size");
  state.add_global_memory_writes<T>(elements);

  size_t tmp_size;
  cub::DeviceScan::InclusiveScan(nullptr,
                                 tmp_size,
                                 input.cbegin(),
                                 output.begin(),
                                 complex_op<OperationsCount>(),
                                 static_cast<int>(input.size()));
  thrust::device_vector<nvbench::uint8_t> tmp(tmp_size);

  state.exec([&input, &output, &tmp](nvbench::launch &launch) {
    std::size_t temp_size = tmp.size(); // need an lvalue
    cub::DeviceScan::InclusiveScan(thrust::raw_pointer_cast(tmp.data()),
                                  temp_size,
                                  input.cbegin(),
                                  output.begin(),
                                  complex_op<OperationsCount>(),
                                  static_cast<int>(input.size()),
                                  launch.get_stream());
  });
}
using types = nvbench::type_list<nvbench::float32_t>;
using ops = nvbench::enum_type_list<64>;
NVBENCH_BENCH_TYPES(complex_op_bench, NVBENCH_TYPE_AXES(types, ops))
  .set_name("cub::DeviceScan::InclusiveScan")
  .add_int64_power_of_two_axis("Elements", nvbench::range(22, 28, 2));
