#pragma once

#include <nvbench/nvbench.cuh>

enum class sort_direction
{
  ascending,
  descending
};

using directions =
  nvbench::enum_type_list<sort_direction::ascending, sort_direction::descending>;

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


using types = nvbench::type_list<nvbench::uint8_t,
                                 nvbench::uint16_t,
                                 nvbench::uint32_t,
                                 nvbench::uint64_t>;
