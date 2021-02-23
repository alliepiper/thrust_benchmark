#pragma once

#include <nvbench/cuda_call.cuh>
#include <nvbench/detail/device_scope.cuh>
#include <nvbench/types.cuh>

#include <cuda_runtime_api.h>

#include <cstdint> // CHAR_BIT
#include <optional>
#include <string_view>
#include <utility>

// Forward declare nvml types:
struct nvmlDevice_st;
using nvmlDevice_t = nvmlDevice_st *;

namespace nvbench
{

namespace detail
{
int get_ptx_version(int);
} // namespace detail

struct device_info
{
  explicit device_info(int device_id);

  // Mainly used by unit tests:
  device_info(int device_id, cudaDeviceProp prop)
      : m_id{device_id}
      , m_prop{prop}
  {}

  /// @return The device's id on the current system.
  [[nodiscard]] int get_id() const { return m_id; }

  /// @return The name of the device.
  [[nodiscard]] std::string_view get_name() const
  {
    return std::string_view(m_prop.name);
  }

  [[nodiscard]] bool is_active() const
  {
    int id{-1};
    NVBENCH_CUDA_CALL(cudaGetDevice(&id));
    return id == m_id;
  }

  void set_active() const { NVBENCH_CUDA_CALL(cudaSetDevice(m_id)); }

  /// @return The SM version of the current device as (major*100) + (minor*10).
  [[nodiscard]] int get_sm_version() const
  {
    return m_prop.major * 100 + m_prop.minor * 10;
  }

  /// @return The PTX version of the current device
  [[nodiscard]] __forceinline__ int get_ptx_version() const
  {
    return detail::get_ptx_version(m_id);
  }

  /// @return The default clock rate of the SM in Hz.
  [[nodiscard]] std::size_t get_sm_default_clock_rate() const
  { // kHz -> Hz
    return static_cast<std::size_t>(m_prop.clockRate * 1000);
  }

  /// @return The number of physical streaming multiprocessors on this device.
  [[nodiscard]] int get_number_of_sms() const
  {
    return m_prop.multiProcessorCount;
  }

  /// @return The maximum number of resident blocks per SM.
  [[nodiscard]] int get_max_blocks_per_sm() const
  {
    return m_prop.maxBlocksPerMultiProcessor;
  }

  /// @return The maximum number of resident threads per SM.
  [[nodiscard]] int get_max_threads_per_sm() const
  {
    return m_prop.maxThreadsPerMultiProcessor;
  }

  /// @return The maximum number of threads per block.
  [[nodiscard]] int get_max_threads_per_block() const
  {
    return m_prop.maxThreadsPerBlock;
  }

  /// @return The number of registers per SM.
  [[nodiscard]] int get_registers_per_sm() const
  {
    return m_prop.regsPerMultiprocessor;
  }

  /// @return The number of registers per block.
  [[nodiscard]] int get_registers_per_block() const
  {
    return m_prop.regsPerBlock;
  }

  /// @return The total number of bytes available in global memory.
  [[nodiscard]] std::size_t get_global_memory_size() const
  {
    return m_prop.totalGlobalMem;
  }

  struct memory_info
  {
    std::size_t bytes_free;
    std::size_t bytes_total;
  };

  /// @return The size and usage of this device's global memory.
  [[nodiscard]] memory_info get_global_memory_usage() const;

  /// @return The peak clock rate of the global memory bus in Hz.
  [[nodiscard]] std::size_t get_global_memory_bus_peak_clock_rate() const
  { // kHz -> Hz
    return static_cast<std::size_t>(m_prop.memoryClockRate) * 1000;
  }

  /// @return The width of the global memory bus in bits.
  [[nodiscard]] int get_global_memory_bus_width() const
  {
    return m_prop.memoryBusWidth;
  }

  //// @return The global memory bus bandwidth in bytes/sec.
  [[nodiscard]] std::size_t get_global_memory_bus_bandwidth() const
  { // 2 is for DDR, CHAR_BITS to convert bus_width to bytes.
    return 2 * this->get_global_memory_bus_peak_clock_rate() *
           (this->get_global_memory_bus_width() / CHAR_BIT);
  }

  /// @return The size of the L2 cache in bytes.
  [[nodiscard]] std::size_t get_l2_cache_size() const
  {
    return static_cast<std::size_t>(m_prop.l2CacheSize);
  }

  /// @return The available amount of shared memory in bytes per SM.
  [[nodiscard]] std::size_t get_shared_memory_per_sm() const
  {
    return m_prop.sharedMemPerMultiprocessor;
  }

  /// @return The available amount of shared memory in bytes per block.
  [[nodiscard]] std::size_t get_shared_memory_per_block() const
  {
    return m_prop.sharedMemPerBlock;
  }

  /// @return True if ECC is enabled on this device.
  [[nodiscard]] bool get_ecc_state() const { return m_prop.ECCEnabled; }

  /// @return A cached copy of the device's cudaDeviceProp.
  [[nodiscard]] const cudaDeviceProp &get_cuda_device_prop() const
  {
    return m_prop;
  }

  struct perf_state
  {
    /// Device throttle reason flags:
    enum class throttle_flag : nvbench::int64_t
    { // Flags follow nvml definitions
      none                = 0x0ll,
      idle                = 0x1ll,
      app_clocks          = 0x2ll,
      sw_power_cap        = 0x4ll,
      hw_slowdown         = 0x8ll,
      sync_boost          = 0x10ll,
      sw_thermal_slowdown = 0x20ll,
      hw_thermal_slowdown = 0x40ll,
      hw_power_brake      = 0x80ll,
      display_clocks      = 0x100ll
    };

    std::optional<throttle_flag> throttle_reason;

    /// The current actual clock speed in Hz. @{
    std::optional<std::size_t> clock_sm;
    std::optional<std::size_t> clock_mem;
    /// @}

    /// The current application clock setting in Hz. @{
    std::optional<std::size_t> application_clock_sm;
    std::optional<std::size_t> application_clock_mem;
    /// @}

    /// The current device utilitization rates, as a percentage of peak. @{
    std::optional<std::size_t> utilization_sm;
    std::optional<std::size_t> utilization_mem;

    /// True if auto_boosted_clocks are enabled.
    std::optional<bool> auto_boosted_clocks_enabled{};

    /// The current temperature of the device in celsius.
    std::optional<std::size_t> temp_current;

    /// Threshold temperature. @{
    std::optional<std::size_t> temp_thresh_mem_max;
    std::optional<std::size_t> temp_thresh_sm_max;
    std::optional<std::size_t> temp_thresh_hw_slowdown;
    std::optional<std::size_t> temp_thresh_hw_shutdown;
    /// @}

    /// Current power usage in watts.
    std::optional<std::size_t> power_usage;

    /// Power state. Integer between 0 (max perf) and 15(min perf).
    std::optional<int> power_state;
  };

  [[nodiscard]] bool operator==(const device_info &o) const
  {
    return m_id == o.m_id;
  }
  [[nodiscard]] bool operator!=(const device_info &o) const
  {
    return m_id != o.m_id;
  }

private:
  int m_id{};
  cudaDeviceProp m_prop{};
  std::optional<nvmlDevice_t> m_nvml_device{};
};

// get_ptx_version implementation; this needs to stay in the header so it will
// pick up the downstream project's compilation settings.
// TODO this is fragile and will break when called from any library
// translation unit.
namespace detail
{
// Templated to workaround ODR issues since __global__functions cannot be marked
// inline.
template <typename>
__global__ void noop_kernel()
{}

inline const auto noop_kernel_ptr = &noop_kernel<void>;

[[nodiscard]] inline int get_ptx_version(int dev_id)
{
  nvbench::detail::device_scope _{dev_id};
  cudaFuncAttributes attr{};
  NVBENCH_CUDA_CALL(
    cudaFuncGetAttributes(&attr, nvbench::detail::noop_kernel_ptr));
  return attr.ptxVersion * 10;
}
} // namespace detail

} // namespace nvbench
