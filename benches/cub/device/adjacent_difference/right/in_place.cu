#include <nvbench/nvbench.cuh>

#include <thrust/adjacent_difference.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

template <typename T>
static void basic(nvbench::state &state, nvbench::type_list<T>)
{
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements"));

  thrust::device_vector<T> data(elements, T(42));

  T *d_data = thrust::raw_pointer_cast(data.data());

  std::uint8_t *d_temp_storage{};
  std::size_t temp_storage_bytes{};

  cub::DeviceAdjacentDifference::SubtractRight(d_temp_storage,
                                               temp_storage_bytes,
                                               d_data,
                                               elements);

  thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements, "Size");
  state.add_global_memory_writes<T>(elements);

  state.exec([&](nvbench::launch &launch) {
    std::size_t temp_size = temp_storage.size(); // need an lvalue
    cub::DeviceAdjacentDifference::SubtractRight(d_temp_storage,
                                                 temp_size,
                                                 d_data,
                                                 elements,
                                                 cub::Difference{},
                                                 launch.get_stream());
  });
}

using types = nvbench::type_list<nvbench::uint8_t,
                                 nvbench::uint16_t,
                                 nvbench::uint32_t,
                                 nvbench::uint64_t,
                                 nvbench::float32_t,
                                 nvbench::float64_t>;

NVBENCH_BENCH_TYPES(basic, NVBENCH_TYPE_AXES(types))
  .set_name("cub::DeviceAdjacentDifference::SubtractRight (basic)")
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 2));
