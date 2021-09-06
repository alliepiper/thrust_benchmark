#include <nvbench/nvbench.cuh>

#include <type_traits>

#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/unique.h>

#include <tbm/range_generator.cuh>

template <typename T,
          typename OperationT>
void basic(nvbench::state &state, nvbench::type_list<T, OperationT>)
{
  const auto elements =
    static_cast<std::size_t>(state.get_int64("Elements"));

  OperationT operation{elements * sizeof(T)};

  thrust::device_vector<T> input(elements);
  thrust::device_vector<T> result(elements);

  state.add_element_count(elements);

  const auto pattern = state.get_string("Pattern");
  if (pattern == "unique")
  {
    thrust::sequence(input.begin(), input.end());

    if (elements > std::numeric_limits<T>::max())
    {
      thrust::sort(input.begin(), input.end());
    }
  }
  else if (pattern == "const")
  {
    thrust::fill(input.begin(), input.end(), T{2});
  }
  else if (pattern == "random")
  {
    auto random_input =
      tbm::make_range_generator<T,
                                tbm::iterator_style::pointer,
                                tbm::data_pattern::random>(elements);

    thrust::copy(random_input.cbegin(), random_input.cend(), input.begin());
    thrust::sort(input.begin(), input.end());
  }

  {
    const std::size_t unique_items = thrust::distance(
      result.begin(),
      thrust::unique_copy(input.begin(), input.end(), result.begin()));

    state.add_global_memory_reads<T>(elements);
    state.add_global_memory_writes<T>(unique_items);

    state.add_global_memory_reads(operation.extra_bytes_to_read());
    state.add_global_memory_writes(
      operation.template extra_bytes_to_write<T>(unique_items));
  }

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch &launch) {
    const auto policy = thrust::device.on(launch.get_stream());

    operation(policy, input.begin(), input.end(), result.begin());
  });
}

using types = nvbench::type_list<nvbench::uint8_t,
                                 nvbench::uint16_t,
                                 nvbench::uint32_t,
                                 nvbench::uint64_t>;
