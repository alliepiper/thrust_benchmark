#pragma once

#include <tbm/range_generator.cuh>

#include <nvbench/nvbench.cuh>

// This is not public API because it depends on fmtlib.
// FIXME: Add a similar macro to tbm::
#include <nvbench/detail/throw.cuh>

// All radix sortable types
using all_key_types = nvbench::type_list<bool,
                                         nvbench::uint8_t,
                                         nvbench::uint16_t,
                                         nvbench::uint32_t,
                                         nvbench::uint64_t,
                                         nvbench::int8_t,
                                         nvbench::int16_t,
                                         nvbench::int32_t,
                                         nvbench::int64_t,
                                         // FIXME: thrust/random doesn't handle
                                         // half
                                         // __half,
                                         nvbench::float32_t,
                                         nvbench::float64_t>;

// Uniquely sized radix sortable types
using unique_size_key_types = nvbench::type_list<nvbench::uint8_t,
                                                 nvbench::uint16_t,
                                                 nvbench::uint32_t,
                                                 nvbench::uint64_t>;

// Very small set of representative key types for minimal testing.
using common_key_types =
  nvbench::type_list<nvbench::uint8_t, nvbench::int32_t, nvbench::float64_t>;

// Value types don't really matter much, they just get permuted. Only benchmark
// with a 32-bit payload.
using value_types = nvbench::type_list<nvbench::uint32_t>;

// Data pattern types:
using random_input   = nvbench::enum_type_list<tbm::data_pattern::random>;
using constant_input = nvbench::enum_type_list<tbm::data_pattern::constant>;

enum class sort_direction
{
  ascending,
  descending
};

using ascending_sort  = nvbench::enum_type_list<sort_direction::ascending>;
using descending_sort = nvbench::enum_type_list<sort_direction::descending>;

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  sort_direction,
  [](sort_direction direction) {
    switch (direction)
    {
      case sort_direction::ascending:
        return "Ascend";
      case sort_direction::descending:
        return "Descend";
      default:
        break;
    }
    NVBENCH_THROW(std::runtime_error, "{}", "Unknown sort_direction");
  },
  // Don't need descriptions:
  [](sort_direction) { return std::string{}; })
