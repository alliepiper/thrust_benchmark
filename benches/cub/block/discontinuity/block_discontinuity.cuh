#pragma once

#include <nvbench/detail/throw.cuh>
#include <nvbench/nvbench.cuh>

#include <thrust/device_vector.h>
#include <thrust/sequence.h>
// Why is this in detail?
#include <thrust/detail/raw_pointer_cast.h>

#include <cub/block/block_discontinuity.cuh>


constexpr int iterations = 10;
__constant__ int input_data[iterations];


template <typename T,
          int ItemsPerThread>
__device__ void fill_thread_data(int pivot,
                                 T (&thread_data)[ItemsPerThread],
                                 const unsigned int linear_id)
{
  for (int i = 0; i < ItemsPerThread; i++)
  {
    thread_data[i] = pivot * static_cast<T>(linear_id + i);
  }
}


template <typename T,
          int ItemsPerThread>
__device__ void do_not_optimize(int *output,
                                T (&thread_data)[ItemsPerThread],
                                const unsigned int linear_id)
{
  int count = 0;
  for (int i = 0; i < ItemsPerThread; i++)
  {
    if (thread_data[i])
    {
      count++;
    }
  }

  if (count > 42)
  {
    atomicAdd(output + linear_id, 1);
  }
}

// This kernel is used to check that runtime of BlockDiscontinuity methods
// exceeds wrapper overhead
template <typename T, int ItemsPerThread>
__global__ void kernel_reference(int *output)
{
  const unsigned int linear_id = threadIdx.x + blockIdx.x * blockDim.x;
  T thread_data[ItemsPerThread];

  for (int pivot: input_data)
  {
    fill_thread_data(pivot, thread_data, linear_id);
    do_not_optimize(output, thread_data, linear_id);
  }
}

template <typename T,
          typename OperationType,
          int ThreadsInBlock,
          int ItemsPerThread>
__global__ void kernel(int *output)
{
  const unsigned int linear_id = threadIdx.x + blockIdx.x * blockDim.x;

  bool flags[ItemsPerThread];
  T thread_data[ItemsPerThread];

  using BlockDiscontinuity = cub::BlockDiscontinuity<T, ThreadsInBlock>;

  __shared__ typename BlockDiscontinuity::TempStorage temp_storage;

  BlockDiscontinuity discontinuity(temp_storage);

  for (int pivot: input_data)
  {
    fill_thread_data(pivot, thread_data, linear_id);
    OperationType()(discontinuity, flags, thread_data);
    do_not_optimize(output, flags, linear_id);
  }
}

template <typename T,
          typename OperationType,
          int ThreadsInBlock,
          int ItemsPerThread>
static void bench(nvbench::state &state,
                  nvbench::type_list<T,
                                     OperationType,
                                     nvbench::enum_type<ThreadsInBlock>,
                                     nvbench::enum_type<ItemsPerThread>>)
{
  const auto blocks_in_grid = 524288;

  std::vector<int> h_data(iterations);
  cudaMemcpyToSymbol(input_data, h_data.data(), sizeof(int) * iterations);

  state.add_element_count(blocks_in_grid);
  state.add_global_memory_reads<T>(blocks_in_grid);

  const bool reference_mode = state.get_string("Mode") == "Reference";

  state.exec([&](nvbench::launch &launch) {
    /**
     * Kernel below reads zeroes and performs BlockDiscontinuity operation.
     * The result is only written if there are at least 42 heads within the
     * result. Because the data is filled with zeroes, there is only one head.
     * Therefore, there is no need in allocating output array. This trick is
     * done to prevent the compiler from optimizing the code.
     */
    int *output = nullptr;

    if (reference_mode)
    {
      kernel_reference<T, ItemsPerThread>
        <<<blocks_in_grid, ThreadsInBlock, 0, launch.get_stream()>>>(output);
    }
    else
    {
      kernel<T, OperationType, ThreadsInBlock, ItemsPerThread>
        <<<blocks_in_grid, ThreadsInBlock, 0, launch.get_stream()>>>(output);
    }
  });
}

inline std::vector<std::string> type_axis_names()
{
  return {"T", "Op", "ThreadsInBlock", "ItemsPerThread"};
}

using types = nvbench::type_list<nvbench::uint32_t, nvbench::uint64_t>;
using threads_in_block = nvbench::enum_type_list<128, 512, 1024>;
using items_per_thread = nvbench::enum_type_list<1, 4>;
