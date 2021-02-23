#pragma once

// This file contains inline statics, and should not be included more than once.
// It just contains nvml utility code used by device_info.cu.
#ifndef NVBENCH_NVML_UTILITY_GUARD
#error "This is a private implementation header for device_info.cu. " \
       "Do not include it directly."
#endif // NVBENCH_NVML_UTILITY_GUARD

#include <nvbench/detail/throw.cuh>

#include <fmt/format.h>

#include <cuda_runtime_api.h>

#include <nvml.h>

#include <mutex> // for call_once
#include <optional>

#define NVBENCH_NVML_CALL(call)                                                \
  do                                                                           \
  {                                                                            \
    const auto _retval = call;                                                 \
    if (_retval != NVML_SUCCESS)                                               \
    {                                                                          \
      NVBENCH_THROW(std::runtime_exception,                                    \
                    "NVML call failed: {}\n\t{}",                              \
                    #call,                                                     \
                    nvmlErrorString(_rr));                                     \
    }                                                                          \
  } while (false)

#define NVBENCH_NVML_CALL_NULLOPT(call)                                        \
  do                                                                           \
  {                                                                            \
    const auto _retval = call;                                                 \
    if (_retval != NVML_SUCCESS)                                               \
    {                                                                          \
      return std::nullopt;                                                     \
    }                                                                          \
  } while (false)

namespace nvbench::internal::nvml
{

/*!
 * RAII context for nvml. Initializes at startup, shuts down at exit.
 * Evaluates to "true" if initialized:
 *
 * ```
 * if (nvbench::internal::nvml::context) { ... }
 * ```
 * @{
 */
struct context_t
{
  context_t()
      : m_initialized{nvmlInit() == NVML_SUCCESS}
  {}
  ~context_t()
  {
    if (m_initialized)
    {
      nvmlShutdown();
    }
  }

  context_t(const context_t &) = delete;
  context_t(context_t &&)      = delete;
  context_t &operator=(const context_t &) = delete;
  context_t &operator=(context_t &&) = delete;

  operator bool() const { return m_initialized; }

private:
  bool m_initialized;
};
static inline context_t context{};
/*!@}*/

std::optional<nvmlDevice_t>
cudaDevicePropToNvmlDevice(const cudaDeviceProp &prop)
{
  auto pci_id = fmt::format("cudaDeviceProp: {:08d}:{:02d}:{:02d}.0\n",
                            prop.pciDomainID,
                            prop.pciBusID,
                            prop.pciDeviceID);
  nvmlDevice_t device{};
  NVBENCH_NVML_CALL_NULLOPT(
    nvmlDeviceGetHandleByPciBusId(pci_id.c_str(), &device));
  return device;
}

} // namespace nvbench::internal::nvml
