#include <nvbench/nvbench.cuh>

#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/merge.h>

#include <tbm/range_generator.cuh>

template <typename T>
void merge(nvbench::state &state,
           nvbench::type_list<T>)
{
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements"));
  const auto sets_size_ratio =
    static_cast<std::size_t>(state.get_int64("InputsSizeRatio"));
  const auto elements_in_first_input = static_cast<std::size_t>(
    static_cast<double>(sets_size_ratio * elements) / 100.0f);
  const auto elements_in_second_input = elements - elements_in_first_input;

  thrust::device_vector<T> first_input(elements_in_first_input);
  thrust::device_vector<T> second_input(elements_in_second_input);
  thrust::device_vector<T> result(elements);


  const auto pattern = state.get_string("Pattern");
  if (pattern == "order")
  {
    thrust::fill(first_input.begin(), first_input.end(), T{0});
    thrust::fill(second_input.begin(), second_input.end(), T{1});
  }
  else if (pattern == "match")
  {
    if (elements_in_first_input > elements_in_second_input)
    {
      thrust::sequence(first_input.begin(), first_input.end());
      thrust::sort(first_input.begin(), first_input.end());
      thrust::copy_n(first_input.begin(),
                     elements_in_second_input,
                     second_input.begin());
    }
    else
    {
      thrust::sequence(second_input.begin(), second_input.end());
      thrust::sort(second_input.begin(), second_input.end());
      thrust::copy_n(second_input.begin(),
                     elements_in_first_input,
                     first_input.begin());
    }
  }
  else if (pattern == "random")
  {
    auto random_input =
      tbm::make_range_generator<T,
                                tbm::iterator_style::pointer,
                                tbm::data_pattern::random>(elements);

    thrust::copy(random_input.cbegin(),
                 random_input.cbegin() + elements_in_first_input,
                 first_input.begin());

    thrust::copy(random_input.cbegin() + elements_in_first_input,
                 random_input.cend(),
                 second_input.begin());

    thrust::sort(first_input.begin(), first_input.end());
    thrust::sort(second_input.begin(), second_input.end());
  }

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements);
  state.add_global_memory_writes<T>(elements);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch &launch) {
    const auto policy = thrust::device.on(launch.get_stream());

    thrust::merge(policy,
                  first_input.begin(),
                  first_input.end(),
                  second_input.begin(),
                  second_input.end(),
                  result.begin());
  });
}

using types = nvbench::type_list<nvbench::int8_t,
                                 nvbench::int16_t,
                                 nvbench::int32_t,
                                 nvbench::int64_t>;

NVBENCH_BENCH_TYPES(merge, NVBENCH_TYPE_AXES(types))
  .set_name("thrust::merge")
  .add_int64_axis("InputsSizeRatio", nvbench::range(1, 100, 24))
  .add_int64_power_of_two_axis("Elements", nvbench::range(20, 30, 2))
  .add_string_axis("Pattern", {"order", "match", "random"});
