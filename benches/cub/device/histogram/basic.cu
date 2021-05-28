#include <nvbench/nvbench.cuh>

#include <thrust/device_vector.h>
#include <thrust/sequence.h>
// Why is this in detail?
#include <thrust/detail/raw_pointer_cast.h>

#include <cub/device/device_histogram.cuh>

#include <tbm/range_generator.cuh>

template <typename T, int BinsCount, tbm::data_pattern Pattern>
static void basic(nvbench::state &state,
                  nvbench::type_list<T,
                                     nvbench::enum_type<BinsCount>,
                                     nvbench::enum_type<Pattern>>)
{
  // Compilation is broken for size_t, why?
  const auto elements = static_cast<int>(state.get_int64("Elements"));

  auto input =
    tbm::make_range_generator<T, tbm::iterator_style::pointer, Pattern>(
      elements, BinsCount);

  const int num_bins = BinsCount;
  const int num_levels = num_bins + 1;

  T lower_level = 0;
  T upper_level = elements;

  if (Pattern == tbm::data_pattern::modulo_sequence)
  {
    upper_level = BinsCount;
  }

  thrust::device_vector<int> histogram(num_bins);

  size_t tmp_size;
  cub::DeviceHistogram::HistogramEven(nullptr,
                                      tmp_size,
                                      input.cbegin(),
                                      thrust::raw_pointer_cast(histogram.data()),
                                      num_levels,
                                      lower_level,
                                      upper_level,
                                      elements);

  thrust::device_vector<nvbench::uint8_t> tmp(tmp_size);

  state.exec([&](nvbench::launch &launch) {
    std::size_t temp_size = tmp.size(); // need an lvalue
    cub::DeviceHistogram::HistogramEven(thrust::raw_pointer_cast(tmp.data()),
                                        temp_size,
                                        input.cbegin(),
                                        thrust::raw_pointer_cast(histogram.data()),
                                        num_levels,
                                        lower_level,
                                        upper_level,
                                        elements);
  });
}

// Column names for type axes:
inline std::vector<std::string> histogram_type_axis_names()
{
  return {"T", "BinsCount", "Pattern"};
}

using bins = nvbench::enum_type_list<2, 2048, 2097152>;
using types =
  nvbench::type_list<nvbench::uint32_t, nvbench::float32_t, nvbench::uint64_t>;

using all_input_data_patterns =
  nvbench::enum_type_list<tbm::data_pattern::sequence,
                          tbm::data_pattern::modulo_sequence,
                          tbm::data_pattern::constant,
                          tbm::data_pattern::random>;

NVBENCH_BENCH_TYPES(basic,
                    NVBENCH_TYPE_AXES(types, bins, all_input_data_patterns))
  .set_name("cub::DeviceHistogram::HistogramEven")
  .set_type_axes_names(histogram_type_axis_names())
  .add_int64_power_of_two_axis("Elements", nvbench::range(22, 28, 2));
