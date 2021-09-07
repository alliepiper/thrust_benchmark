#pragma once

#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/random.h>
#include <thrust/fill.h>

#include <tbm/range_generator.cuh>

thrust::device_vector<nvbench::uint32_t> gen_offsets(int segment_size,
                                                     int elements)
{
  const int num_segments = (elements + segment_size - 1) / segment_size;
  thrust::device_vector<nvbench::uint32_t> offsets(num_segments + 1);
  thrust::fill(offsets.begin(),
               offsets.end(),
               static_cast<nvbench::uint32_t>(segment_size));
  thrust::exclusive_scan(offsets.begin(), offsets.end(), offsets.begin());
  offsets[num_segments] = elements;
  return offsets;
}


template <typename T>
struct SegmentSizeLimiter
{
  T out_max;

  SegmentSizeLimiter(T out_max)
    : out_max(out_max)
  {}

  __device__ T operator()(T in)
  {
    T guarded_value = in % out_max;
    return guarded_value < 1 ? 1 : guarded_value;
  }
};


thrust::device_vector<nvbench::uint32_t>
gen_random(nvbench::uint32_t max_segment_size, int elements)
{
  thrust::device_vector<nvbench::uint32_t> offsets(elements);

  auto random_input =
    tbm::make_range_generator<nvbench::uint32_t,
      tbm::iterator_style::pointer,
      tbm::data_pattern::random>(elements);

  thrust::transform(random_input.cbegin(),
                    random_input.cend(),
                    offsets.begin(),
                    SegmentSizeLimiter<nvbench::uint32_t>{max_segment_size});

  thrust::exclusive_scan(offsets.begin(), offsets.end(), offsets.begin());
  const auto num_segments = thrust::distance(
    offsets.begin(),
    thrust::lower_bound(offsets.begin(),
                        thrust::is_sorted_until(offsets.begin(), offsets.end()),
                        static_cast<nvbench::uint32_t>(elements)));

  offsets.resize(num_segments + 1);
  offsets[num_segments] = elements;

  return offsets;
}


thrust::device_vector<nvbench::uint32_t> gen_offsets(const std::string &pattern,
                                                     int elements)
{
  if (pattern == "small")
  {
    return gen_offsets(4, elements);
  }
  else if (pattern == "large")
  {
    return gen_offsets(256 * 1024, elements);
  }

  return gen_random(16 * 1024, elements);
}
