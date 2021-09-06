#include <nvbench/nvbench.cuh>

#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>

#include <tbm/range_generator.cuh>

std::size_t get_needles_num(std::size_t haystack_size,
                            std::size_t needles_ratio)
{
  return needles_ratio *
         static_cast<std::size_t>(static_cast<double>(haystack_size) / 100.0);
}

template <typename T>
thrust::device_vector<T> gen_haystack(std::size_t elements)
{
  auto random_input =
    tbm::make_range_generator<T,
                              tbm::iterator_style::pointer,
                              tbm::data_pattern::random>(elements);

  thrust::device_vector<T> haystack(random_input.cbegin(), random_input.cend());
  thrust::sort(haystack.begin(), haystack.end());
  return haystack;
}

template <typename T,
          typename OperationT>
void search(nvbench::state &state,
            nvbench::type_list<T, OperationT>)
{
  const auto haystack_size =
    static_cast<std::size_t>(state.get_int64("Elements"));
  const auto needles_ratio =
    static_cast<std::size_t>(state.get_int64("NeedlesRatio"));

  const auto needles_num = get_needles_num(haystack_size, needles_ratio);

  thrust::device_vector<T> haystack = gen_haystack<T>(haystack_size);
  thrust::device_vector<std::size_t> result(needles_num);
  thrust::device_vector<T> needles(needles_num);

  OperationT operation{};

  const auto pattern = state.get_string("Pattern");
  if (pattern == "sequence")
  {
    thrust::sequence(needles.begin(), needles.end());
  }
  else if (pattern == "uniform")
  {
    const auto step = static_cast<T>(haystack_size / needles_num);
    thrust::sequence(needles.begin(), needles.end(), T{}, step);
  }
  else if (pattern == "random")
  {
    needles = haystack;

    thrust::default_random_engine re;
    thrust::shuffle(needles.begin(), needles.end(), re);
    needles.resize(needles_num);
  }

  state.collect_dram_throughput();
  state.collect_loads_efficiency();

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch &launch) {
    const auto policy = thrust::device.on(launch.get_stream());
    operation(policy,
              haystack.begin(),
              haystack.end(),
              needles.begin(),
              needles.end(),
              result.begin());
  });
}

using types = nvbench::type_list<nvbench::uint32_t>;
