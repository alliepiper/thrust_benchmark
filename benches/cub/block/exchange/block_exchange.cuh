#pragma once

#include <nvbench/detail/throw.cuh>
#include <nvbench/nvbench.cuh>

#include <thrust/device_vector.h>
#include <thrust/sequence.h>
// Why is this in detail?
#include <thrust/detail/raw_pointer_cast.h>

#include <cub/block/block_exchange.cuh>

template <typename T, int ItemsPerThread>
__device__ void fill_thread_data(const T *input,
                                 T (&thread_data)[ItemsPerThread],
                                 const unsigned int linear_id)
{
  const T block_input = input[blockIdx.x];

  for (int i = 0; i < ItemsPerThread; i++)
  {
    thread_data[i] = block_input * (linear_id + i);
  }
}

template <typename T, int ItemsPerThread>
__device__ void do_not_optimize(int *output,
                                T (&thread_data)[ItemsPerThread],
                                const unsigned int linear_id)
{
  int count = 0;
  for (int i = 0; i < ItemsPerThread; i++)
  {
    if (thread_data[i] > T(42))
    {
      count++;
    }
  }

  if (count)
  {
    output[linear_id] = count;
  }
}

// This kernel is used to check that runtime of BlockExchange methods
// exceeds wrapper overhead
template <typename T, int ItemsPerThread>
__global__ void kernel_reference(const T *input, int *output)
{
  const unsigned int linear_id = threadIdx.x + blockIdx.x * blockDim.x;
  T thread_data[ItemsPerThread];

  fill_thread_data(input, thread_data, linear_id);
  do_not_optimize(output, thread_data, linear_id);
}

template <typename T,
          typename OperationType,
          int ThreadsInBlock,
          int ItemsPerThread,
          bool WarpTimeSlicing>
__global__ void kernel(const T *input, int *output)
{
  const unsigned int linear_id = threadIdx.x + blockIdx.x * blockDim.x;
  T thread_data[ItemsPerThread];

  fill_thread_data(input, thread_data, linear_id);

  using BlockExchange =
    cub::BlockExchange<T, ThreadsInBlock, ItemsPerThread, WarpTimeSlicing>;

  __shared__ typename BlockExchange::TempStorage temp_storage;

  // Initialize the shared memory to 0.
  //
  // This isn't normally needed, but otherwise we'll see invalid memory accesses
  // in `do_not_optimize` as a result of the conditional exchanges in
  // `scatter_to_striped_guarded` and `scatter_to_striped_flagged`. The items
  // where `rank[i] == -1` (guarded) or `is_valid[i] == false` (flagged) will
  // copy the contents of (uninitialized) shared memory into `thread_data`.
  //
  // Since `do_not_optimize` expects all elements of `thread_data` to be 0,
  // this will cause the intentionally-unreachable `output[linear_id] = count`
  // branch to be taken. `output` is always a nullptr, thus the invalid write.
  //
  // By explicitly zeroing out temp_storage, we can avoid this issue.
  if (threadIdx.x == 0)
  {
    memset(&temp_storage, 0, sizeof(temp_storage));
  }
  __syncthreads();

  BlockExchange exchange(temp_storage);
  OperationType()(exchange, thread_data);

  do_not_optimize(output, thread_data, linear_id);
}

enum class compute_mode
{
  reference,
  exchange,
  exchange_warp_time_slicing
};

// Column names for type axes:
inline std::vector<std::string> block_exchange_type_axis_names()
{
  return {"T", "Op", "ThreadsInBlock", "ItemsPerThread", "ComputeMode"};
}

template <typename T,
          typename OperationType,
          int ThreadsInBlock,
          int ItemsPerThread,
          compute_mode ComputeMode>
static void bench(nvbench::state &state,
                  nvbench::type_list<T,
                                     OperationType,
                                     nvbench::enum_type<ThreadsInBlock>,
                                     nvbench::enum_type<ItemsPerThread>,
                                     nvbench::enum_type<ComputeMode>>)
{
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements"));

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements);

  thrust::device_vector<T> input(elements);
  const T *d_input = thrust::raw_pointer_cast(input.data());

  state.exec([&](nvbench::launch &launch) {
    /**
     * All kernels below read some data and count the number it elements
     * exceeding 42. The result is only written if there is at least one
     * item with a value greater than 42. Because the data is filled with
     * zeroes, there are no writes to the output array. Therefore, there
     * is no need in allocating it. This trick is done to prevent the
     * compiler from optimizing the code.
     */
    int *output = nullptr;

    if constexpr(compute_mode::reference == ComputeMode)
    {
      kernel_reference<T, ItemsPerThread>
        <<<elements, ThreadsInBlock>>>(d_input, output);
    }
    else if constexpr(compute_mode::exchange == ComputeMode)
    {
      kernel<T, OperationType, ThreadsInBlock, ItemsPerThread, false>
        <<<elements, ThreadsInBlock>>>(d_input, output);
    }
    else if constexpr(compute_mode::exchange_warp_time_slicing == ComputeMode)
    {
      kernel<T, OperationType, ThreadsInBlock, ItemsPerThread, true>
        <<<elements, ThreadsInBlock>>>(d_input, output);
    }
  });
}

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  compute_mode,
  [](compute_mode mode) {
    switch (mode)
    {
      case compute_mode::reference:
        return "Reference";
      case compute_mode::exchange:
        return "Exchange";
      case compute_mode::exchange_warp_time_slicing:
        return "ExchangeWarpTimeSlicing";
      default:
        break;
    }
    NVBENCH_THROW(std::runtime_error, "{}", "Unknown data_pattern");
  },
  [](compute_mode mode) {
    switch (mode)
    {
      case compute_mode::reference:
        return "Load data without block exchange";
      case compute_mode::exchange:
        return "Load data and perform block exchange";
      case compute_mode::exchange_warp_time_slicing:
        return "Load data and perform block exchange with limited shared "
               "memory";
      default:
        break;
    }
    NVBENCH_THROW(std::runtime_error, "{}", "Unknown data_pattern");
  })

using types = nvbench::type_list<nvbench::uint32_t, nvbench::uint64_t>;
using threads_in_block = nvbench::enum_type_list<32, 512, 1024>;
using items_per_thread = nvbench::enum_type_list<1, 2, 4>;
using compute_modes =
  nvbench::enum_type_list<compute_mode::reference,
                          compute_mode::exchange,
                          compute_mode::exchange_warp_time_slicing>;
