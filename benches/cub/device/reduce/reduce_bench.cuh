#pragma once

#include <tbm/range_generator.cuh>

#include <nvbench/nvbench.cuh>

#include <cub/device/device_reduce.cuh>

#include <string>
#include <type_traits>
#include <vector>

// Column names for type axes:
inline std::vector<std::string> reduce_type_axis_names()
{
  return {"T", "ReduceOp", "Input", "Pattern"};
}

// Custom sum implementation to compare with optimized Sum implementations
// (see ./notes.md).
struct custom_sum
{
  template <typename T, typename U>
  __forceinline__ __host__ __device__ auto operator()(T &&t, U &&u)
  {
    return std::forward<T>(t) + std::forward<U>(u);
  }
};

// ReduceOp type axis definitions:
using cub_max_op = nvbench::type_list<cub::Max>;
using sum_ops    = nvbench::type_list<cub::Sum, custom_sum>;

NVBENCH_DECLARE_TYPE_STRINGS(custom_sum, "CustomSum", "");
NVBENCH_DECLARE_TYPE_STRINGS(cub::Sum, "cub::Sum", "");
NVBENCH_DECLARE_TYPE_STRINGS(cub::Max, "cub::Max", "");

// ValueType axis definitions:
using all_value_types = nvbench::type_list<nvbench::uint8_t,
                                           nvbench::int8_t,
                                           nvbench::uint16_t,
                                           nvbench::int16_t,
                                           nvbench::uint32_t,
                                           nvbench::int32_t,
                                           nvbench::float32_t,
                                           nvbench::uint64_t,
                                           nvbench::int64_t,
                                           nvbench::float64_t>;
// Small representative set, used for larger parameter sweeps:
using common_value_types =
  nvbench::type_list<nvbench::uint8_t, nvbench::int32_t, nvbench::float64_t>;

// iterator_style enum_type_axis definitions
using all_input_iter_styles =
  nvbench::enum_type_list<tbm::iterator_style::pointer,
                          tbm::iterator_style::vector,
                          tbm::iterator_style::counting,
                          tbm::iterator_style::constant>;
using pointer_iter_style =
  nvbench::enum_type_list<tbm::iterator_style::pointer>;

// data_pattern enum_type_axis definitions
using all_input_data_patterns =
  nvbench::enum_type_list<tbm::data_pattern::sequence,
                          tbm::data_pattern::constant,
                          tbm::data_pattern::random>;
using sequence_data_pattern =
  nvbench::enum_type_list<tbm::data_pattern::sequence>;

template <typename T,
          typename ReduceOp,
          tbm::iterator_style IterStyle,
          tbm::data_pattern Pattern>
// Only instantiate if the input type/iter/pattern combo is valid:
std::enable_if_t<tbm::range_generator<T, IterStyle, Pattern>::is_valid(), void>
reduce(nvbench::state &state,
       nvbench::type_list<T,
                          ReduceOp,
                          nvbench::enum_type<IterStyle>,
                          nvbench::enum_type<Pattern>>)
{
  // Retrieve axis parameters
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements"));

  // Allocate and initialize data:
  auto input  = tbm::make_range_generator<T, IterStyle, Pattern>(elements);
  auto output = tbm::make_range_generator<T>(1);

  // Give the generators a chance to skip if their runtime config is invalid:
  if (input.should_skip(state) || output.should_skip(state))
  {
    return;
  }

  // Enable throughput calculations and add "Size" column to results.
  state.add_element_count(elements);
  state.add_global_memory_reads(input.get_allocation_size(), "Size");
  state.add_global_memory_writes(output.get_allocation_size());

  // Allocate temporary storage:
  std::size_t temp_size;
  cub::DeviceReduce::Reduce(nullptr,
                            temp_size,
                            input.cbegin(),
                            output.begin(),
                            static_cast<int>(elements),
                            ReduceOp{},
                            T{});
  thrust::device_vector<nvbench::uint8_t> temp(temp_size);
  auto *temp_storage = thrust::raw_pointer_cast(temp.data());

  if constexpr (Pattern == tbm::data_pattern::random)
  { // Need to reset and add a timer for random patterns.
    state.exec(nvbench::exec_tag::timer,
               [&](nvbench::launch &launch, auto &timer) {
                 input.reset();
                 timer.start();
                 cub::DeviceReduce::Reduce(temp_storage,
                                           temp_size,
                                           input.cbegin(),
                                           output.begin(),
                                           static_cast<int>(elements),
                                           ReduceOp{},
                                           T{},
                                           launch.get_stream());
                 timer.stop();
               });
  }
  else
  { // no reset needed for other patterns.
    state.exec([&](nvbench::launch &launch) {
      cub::DeviceReduce::Reduce(temp_storage,
                                temp_size,
                                input.cbegin(),
                                output.begin(),
                                static_cast<int>(elements),
                                ReduceOp{},
                                T{},
                                launch.get_stream());
    });
  }
}

// Skip invalid type/iterator/pattern configurations:
template <typename T,
          typename ReduceOp,
          tbm::iterator_style IterStyle,
          tbm::data_pattern Pattern>
// Only instantiate if the input type/iter/pattern combo is valid:
std::enable_if_t<!tbm::range_generator<T, IterStyle, Pattern>::is_valid(), void>
reduce(nvbench::state &state,
       nvbench::type_list<T,
                          ReduceOp,
                          nvbench::enum_type<IterStyle>,
                          nvbench::enum_type<Pattern>>)
{
  using gen_t = tbm::range_generator<T, IterStyle, Pattern>;
  // Add the skip reason:
  [[maybe_unused]] bool _ = gen_t{}.should_skip(state);
}
