#include <nvbench/nvbench.cuh>

#include <type_traits>

#include <thrust/set_operations.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>

#include <tbm/range_generator.cuh>


template <typename T>
void gen_full_input(thrust::device_vector<T> &large,
                    thrust::device_vector<T> &small)
{
  thrust::sequence(large.begin(), large.end());
  thrust::sort(large.begin(), large.end());
  thrust::copy_n(large.begin(), small.size(), small.begin());
}


template <typename T,
          typename OperationT>
void basic(nvbench::state &state, nvbench::type_list<T, OperationT>)
{
  const auto elements =
    static_cast<std::size_t>(state.get_int64("Elements"));
  const auto sets_size_ratio =
    static_cast<std::size_t>(state.get_int64("SetsSizeRatio"));
  const auto elements_in_B = static_cast<std::size_t>(
    static_cast<double>(sets_size_ratio * elements) / 100.0f);
  const auto elements_in_A = elements - elements_in_B;

  OperationT operation{elements_in_A * sizeof(T), elements_in_B * sizeof(T)};

  thrust::device_vector<T> input_A(elements_in_A);
  thrust::device_vector<T> input_B(elements_in_B);

  state.add_element_count(elements);

  thrust::device_vector<T> result(elements);

  const auto pattern = state.get_string("Pattern");
  if (pattern == "full")
  {
    if (elements_in_A > elements_in_B)
    {
      gen_full_input(input_A, input_B);
    }
    else
    {
      gen_full_input(input_B, input_A);
    }
  }
  else if (pattern == "empty")
  {
    thrust::fill(input_A.begin(), input_A.end(), T{2});
    thrust::fill(input_B.begin(), input_B.end(), T{4});
  }
  else if (pattern == "random")
  {
    auto random_input =
      tbm::make_range_generator<T,
                                tbm::iterator_style::pointer,
                                tbm::data_pattern::random>(elements);

    thrust::copy(random_input.cbegin(),
                 random_input.cbegin() + elements_in_A,
                 input_A.begin());

    thrust::copy(random_input.cbegin() + elements_in_A,
                 random_input.cend(),
                 input_B.begin());

    thrust::sort(input_A.begin(), input_A.end());
    thrust::sort(input_B.begin(), input_B.end());
  }

  state.collect_dram_throughput();
  state.collect_loads_efficiency();

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch &launch) {
    const auto policy = thrust::device.on(launch.get_stream());

    operation(policy,
              input_A.begin(),
              input_A.end(),
              input_B.begin(),
              input_B.end(),
              result.begin());
  });
}


using types = nvbench::type_list<nvbench::uint8_t,
                                 nvbench::uint16_t,
                                 nvbench::uint32_t,
                                 nvbench::uint64_t>;
