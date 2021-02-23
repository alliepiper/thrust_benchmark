#include <nvbench/device_info.cuh>

#include <nvbench/cuda_call.cuh>
#include <nvbench/detail/device_scope.cuh>

#include <cuda_runtime_api.h>

#define NVBENCH_NVML_UTILITY_GUARD
#include <nvbench/internal/nvml_utility.cuh>
#undef NVBENCH_NVML_UTILITY_GUARD

namespace nvbench
{

device_info::memory_info device_info::get_global_memory_usage() const
{
  nvbench::detail::device_scope _{m_id};

  memory_info result{};
  NVBENCH_CUDA_CALL(cudaMemGetInfo(&result.bytes_free, &result.bytes_total));
  return result;
}

device_info::device_info(int id)
    : m_id{id}
{
  NVBENCH_CUDA_CALL(cudaGetDeviceProperties(&m_prop, m_id));
  if (nvbench::internal::nvml::context)
  {
    m_nvml_device = nvbench::internal::nvml::cudaDevicePropToNvmlDevice(m_prop);
  }
}

} // namespace nvbench
