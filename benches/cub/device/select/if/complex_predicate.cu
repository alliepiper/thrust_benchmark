#include <nvbench/nvbench.cuh>

#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
// Why is this in detail?
#include <thrust/detail/raw_pointer_cast.h>
#include <cub/device/device_select.cuh>

#include <tbm/range_generator.cuh>

template <typename T, unsigned int OperationsLeft>
struct helper;

template <typename T>
struct helper<T, 1>
{
  static __device__ bool compute(T lhs, T rhs)
  {
    return lhs < rhs;
  }
};

template <typename T, unsigned int OperationsLeft>
struct helper
{
  static __device__ bool compute(T lhs, T rhs)
  {
    return helper<T, OperationsLeft - 1>::compute(std::sqrt(lhs * lhs),
                                                  rhs);
  }
};

template <typename T, unsigned int OperationsCount>
struct complex_select_op
{
  T m_val {};

  explicit complex_select_op(T val)
      : m_val(val)
  {}

  __device__ bool operator()(const T& val)
  {
    return helper<T, OperationsCount>::compute(val, m_val);
  }
};

template <typename T, int OperationsCount, tbm::data_pattern Pattern>
static void basic(nvbench::state &state,
                  nvbench::type_list<T,
                                     nvbench::enum_type<OperationsCount>,
                                     nvbench::enum_type<Pattern>>)
{
  const auto elements = static_cast<int>(state.get_int64("Elements"));

  auto input =
    tbm::make_range_generator<T, tbm::iterator_style::pointer, Pattern>(
      elements);

  thrust::device_vector<T> output(elements);
  thrust::device_vector<T> num_selected(1);

  complex_select_op<T, OperationsCount> select_op{0.42f};

  auto selected_elements =
    thrust::count_if(thrust::device, input.cbegin(), input.cend(), select_op);

  state.add_element_count(elements);
  state.add_global_memory_reads(input.get_allocation_size());
  state.add_global_memory_writes<T>(selected_elements);

  size_t tmp_size;
  cub::DeviceSelect::If(nullptr,
                        tmp_size,
                        input.cbegin(),
                        thrust::raw_pointer_cast(output.data()),
                        thrust::raw_pointer_cast(num_selected.data()),
                        elements,
                        select_op);

  thrust::device_vector<nvbench::uint8_t> tmp(tmp_size);

  state.exec([&](nvbench::launch &launch) {
    std::size_t temp_size = tmp.size(); // need an lvalue
    cub::DeviceSelect::If(thrust::raw_pointer_cast(tmp.data()),
                          temp_size,
                          input.cbegin(),
                          thrust::raw_pointer_cast(output.data()),
                          thrust::raw_pointer_cast(num_selected.data()),
                          elements,
                          select_op);
  });
}

// Column names for type axes:
inline std::vector<std::string> select_if_type_axis_names()
{
  return {"T", "Op", "Pattern"};
}

using types =
  nvbench::type_list<nvbench::float32_t>;

using ops = nvbench::enum_type_list<128>;

using all_input_data_patterns =
  nvbench::enum_type_list<tbm::data_pattern::sequence,
                          tbm::data_pattern::constant,
                          tbm::data_pattern::random>;

NVBENCH_BENCH_TYPES(basic,
                    NVBENCH_TYPE_AXES(types, ops, all_input_data_patterns))
  .set_name("cub::DeviceSelect::If")
  .set_type_axes_names(select_if_type_axis_names())
  .add_int64_power_of_two_axis("Elements", nvbench::range(22, 28, 2));
