#include <nvbench/nvbench.cuh>

#include <thrust/device_vector.h>
#include <thrust/partition.h>

#include <tbm/range_generator.cuh>

enum class select_op_type
{
  greater_than_middle,
  greater_than_zero,
  even
  // complex predicate case is covered
  // in the thrust/partition/complex_predicate.cu
};

template <typename T>
struct gt_select_op
{
  T m_val {};

  explicit gt_select_op(T val)
      : m_val(val)
  {}

  __device__ bool operator()(const T& val)
  {
    return val > m_val;
  }
};

template <typename T>
struct even_select_op
{
  __device__ bool operator()(const T& val)
  {
    return (val % 2) == 0;
  }
};

template <select_op_type op_type>
struct op_construction_helper;

template <>
struct op_construction_helper<select_op_type::greater_than_middle>
{
  template <typename T>
  static gt_select_op<T> create_select_op(T elements)
  {
    return gt_select_op<T>{static_cast<T>(elements / 2)};
  }
};

template <>
struct op_construction_helper<select_op_type::greater_than_zero>
{
  template <typename T>
  static gt_select_op<T> create_select_op(int /* elements */)
  {
    return gt_select_op<T>{T{}};
  }
};

template <>
struct op_construction_helper<select_op_type::even>
{
  template <typename T>
  static even_select_op<T> create_select_op(int /* elements */)
  {
    return even_select_op<T>{};
  }
};

template <typename T, select_op_type SelectOpType, tbm::data_pattern Pattern>
static void basic(nvbench::state &state,
                  nvbench::type_list<T,
                                     nvbench::enum_type<SelectOpType>,
                                     nvbench::enum_type<Pattern>>)
{
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements"));

  auto stencil =
    tbm::make_range_generator<T, tbm::iterator_style::pointer, Pattern>(
      elements);

  thrust::device_vector<T> input(elements);
  thrust::device_vector<T> output(elements);

  auto select_op =
    op_construction_helper<SelectOpType>::template create_select_op<T>(
      elements);

  state.add_element_count(elements);
  state.add_global_memory_reads(stencil.get_allocation_size() +
                                elements * sizeof(T));
  state.add_global_memory_writes<T>(elements);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch &launch) {
    auto policy = thrust::device.on(launch.get_stream());

    /*
     * Partition allocates temporary storage and copies the input data into it.
     * This allocation obscures the performance of the partition algorithm.
     * That's the main reason to benchmark partition_copy instead of partition.
     */
    thrust::partition_copy(policy,
                           input.cbegin(),
                           input.cend(),
                           stencil.cbegin(),
                           output.begin(),
                           thrust::make_reverse_iterator(output.begin() +
                                                         elements),
                           select_op);
  });
}

// Column names for type axes:
inline std::vector<std::string> select_if_type_axis_names()
{
  return {"T", "Op", "Pattern"};
}

using types = nvbench::type_list<nvbench::uint8_t,
                                 nvbench::uint16_t,
                                 nvbench::uint32_t,
                                 nvbench::uint64_t>;

using ops = nvbench::enum_type_list<select_op_type::greater_than_middle,
                                    select_op_type::greater_than_zero,
                                    select_op_type::even>;

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  select_op_type,
  [](select_op_type select_op) {
    switch (select_op)
    {
      case select_op_type::greater_than_middle:
        return "Mid";
      case select_op_type::greater_than_zero:
        return "Zero";
      case select_op_type::even:
        return "Even";
      default:
        break;
    }
    NVBENCH_THROW(std::runtime_error, "{}", "Unknown data_pattern");
  },
  [](select_op_type select_op) {
    switch (select_op)
    {
      case select_op_type::greater_than_middle:
        return "Return true for elements > Elements/2";
      case select_op_type::greater_than_zero:
        return "Return true for all elements";
      case select_op_type::even:
        return "Return true for even elements";
      default:
        break;
    }
    NVBENCH_THROW(std::runtime_error, "{}", "Unknown data_pattern");
  })

using all_input_data_patterns =
  nvbench::enum_type_list<tbm::data_pattern::sequence,
                          tbm::data_pattern::constant,
                          tbm::data_pattern::random>;

NVBENCH_BENCH_TYPES(basic,
                    NVBENCH_TYPE_AXES(types, ops, all_input_data_patterns))
  .set_name("thrust::partition (stencil)")
  .set_type_axes_names(select_if_type_axis_names())
  .add_int64_power_of_two_axis("Elements", nvbench::range(20, 28, 2));
