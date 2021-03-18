#pragma once

#include "type_lists.cuh"

#include <tbm/range_generator.cuh>

#include <nvbench/nvbench.cuh>

#include <cub/device/device_radix_sort.cuh>

#include <string>
#include <type_traits>
#include <vector>

inline std::vector<std::string> sort_pairs_type_axis_names()
{
  return {"Key", "Value", "Pattern", "Order"};
}

template <typename KeyType,
          typename ValueType,
          tbm::data_pattern Pattern,
          sort_direction SortDirection>
// Enable if the key type/iterator/pattern combo is valid:
std::enable_if_t<
  tbm::range_generator<KeyType, tbm::iterator_style::pointer, Pattern>::is_valid(),
  void>
sort_pairs(nvbench::state &state,
           nvbench::type_list<KeyType,
                              ValueType,
                              nvbench::enum_type<Pattern>,
                              nvbench::enum_type<SortDirection>>)
{
  // Retrieve axis parameters
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements"));
  // How many key bits to radix sort. Valid values are "All" and "Half":
  const auto bits = state.get_string_or_default("Bits", "All");

  // Allocate and initialize data:
  // Hardcoded to pointers since the API doesn't currently support iterators.
  auto input_keys =
    tbm::make_range_generator<KeyType, tbm::iterator_style::pointer, Pattern>(
      elements);
  auto output_keys = tbm::make_range_generator<KeyType>(elements);

  auto input_values  = tbm::make_range_generator<ValueType>(elements);
  auto output_values = tbm::make_range_generator<ValueType>(elements);

  // Give the generators a chance to skip if their runtime config is invalid:
  if (input_keys.should_skip(state) || output_keys.should_skip(state) ||
      input_values.should_skip(state) || output_values.should_skip(state))
  {
    return;
  }

  // Enable throughput calculations and add "Size" column
  // to results.
  state.add_element_count(elements);
  state.add_global_memory_reads(input_keys.get_allocation_size(), "KeysSize");
  state.add_global_memory_reads(input_values.get_allocation_size(),
                                "ValuesSize");
  state.add_global_memory_writes(output_keys.get_allocation_size());
  state.add_global_memory_writes(output_values.get_allocation_size());

  const auto key_bits     = static_cast<int>(sizeof(KeyType) * 8);
  const auto bits_divisor = bits == "Half" ? 2 : 1;

  // Prepare kernel args:
  const auto *keys_in   = input_keys.cbegin();
  auto *keys_out        = output_keys.begin();
  const auto *values_in = input_values.cbegin();
  auto *values_out      = output_values.begin();
  const auto begin_bit  = int{0};
  const auto end_bit    = key_bits / bits_divisor;

  // Allocate temporary storage:
  std::size_t temp_size;
  if constexpr (SortDirection == sort_direction::ascending)
  {
    NVBENCH_CUDA_CALL(
      cub::DeviceRadixSort::SortPairs(nullptr,
                                      temp_size,
                                      keys_in,
                                      keys_out,
                                      values_in,
                                      values_out,
                                      static_cast<int>(elements),
                                      begin_bit,
                                      end_bit));
  }
  else
  {
    NVBENCH_CUDA_CALL(
      cub::DeviceRadixSort::SortPairsDescending(nullptr,
                                                temp_size,
                                                keys_in,
                                                keys_out,
                                                values_in,
                                                values_out,
                                                static_cast<int>(elements),
                                                begin_bit,
                                                end_bit));
  }
  thrust::device_vector<nvbench::uint8_t> temp(temp_size);
  auto *temp_storage = thrust::raw_pointer_cast(temp.data());

  // Call a different overload of `exec` depending on whether or not we need
  // a timer to cut out a reset of the input.
  if constexpr (!input_keys.needs_reset() &&
                // Constant input data won't need to be reset after sorting:
                Pattern == tbm::data_pattern::constant)
  {
    state.exec([&](nvbench::launch &launch) {
      if constexpr (SortDirection == sort_direction::ascending)
      {
        NVBENCH_CUDA_CALL(
          cub::DeviceRadixSort::SortPairs(temp_storage,
                                          temp_size,
                                          keys_in,
                                          keys_out,
                                          values_in,
                                          values_out,
                                          static_cast<int>(elements),
                                          begin_bit,
                                          end_bit,
                                          launch.get_stream()));
      }
      else // descending
      {
        NVBENCH_CUDA_CALL(
          cub::DeviceRadixSort::SortPairsDescending(temp_storage,
                                                    temp_size,
                                                    keys_in,
                                                    keys_out,
                                                    values_in,
                                                    values_out,
                                                    static_cast<int>(elements),
                                                    begin_bit,
                                                    end_bit,
                                                    launch.get_stream()));
      }
    });
  }
  else // needs timer/reset:
  {
    state.exec(nvbench::exec_tag::timer,
               [&](nvbench::launch &launch, auto &timer) {
                 // Do reset:
                 input_keys.reset();
                 keys_in = input_keys.cbegin();

                 timer.start();
                 if constexpr (SortDirection == sort_direction::ascending)
                 {
                   NVBENCH_CUDA_CALL(
                     cub::DeviceRadixSort::SortPairs(temp_storage,
                                                     temp_size,
                                                     keys_in,
                                                     keys_out,
                                                     values_in,
                                                     values_out,
                                                     static_cast<int>(elements),
                                                     begin_bit,
                                                     end_bit,
                                                     launch.get_stream()));
                 }
                 else // descending
                 {
                   NVBENCH_CUDA_CALL(cub::DeviceRadixSort::SortPairsDescending(
                     temp_storage,
                     temp_size,
                     keys_in,
                     keys_out,
                     values_in,
                     values_out,
                     static_cast<int>(elements),
                     begin_bit,
                     end_bit,
                     launch.get_stream()));
                 }
                 timer.stop();
               });
  }
}

// Skip invalid type/iterator/pattern configurations:
template <typename KeyType,
          typename ValueType,
          tbm::data_pattern Pattern,
          sort_direction SortDirection>
std::enable_if_t<!tbm::range_generator<KeyType,
                                       tbm::iterator_style::pointer,
                                       Pattern>::is_valid(),
                 void>
sort_pairs(nvbench::state &state,
           nvbench::type_list<KeyType,
                              ValueType,
                              nvbench::enum_type<Pattern>,
                              nvbench::enum_type<SortDirection>>)
{
  using gen_t =
    tbm::range_generator<KeyType, tbm::iterator_style::pointer, Pattern>;
  // Add the skip reason:
  [[maybe_unused]] bool _ = gen_t{}.should_skip(state);
}
