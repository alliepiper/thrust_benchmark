#include <nvbench/nvbench.cuh>

#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>
#include <thrust/sort.h>

template <typename T>
void basic(nvbench::state &state, nvbench::type_list<T>)
{
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements"));

  thrust::device_vector<T> data(elements);
  thrust::sequence(data.begin(), data.end());

  thrust::default_random_engine rng;

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements);
  state.add_global_memory_writes<T>(elements);

  using namespace nvbench::exec_tag;
  state.exec(timer | sync, // This benchmark needs a timer and syncs internally
             [&rng, &data](nvbench::launch &launch, auto &timer) {
               const auto policy = thrust::device.on(launch.get_stream());
               thrust::shuffle(policy, data.begin(), data.end(), rng);
               timer.start();
               thrust::sort(policy, data.begin(), data.end());
               timer.stop();
             });
}
using types = nvbench::type_list<nvbench::uint8_t,
                                 nvbench::uint16_t,
                                 nvbench::uint32_t,
                                 nvbench::uint64_t,
                                 nvbench::float32_t,
                                 nvbench::float64_t>;
NVBENCH_BENCH_TYPES(basic, NVBENCH_TYPE_AXES(types))
  .set_name("thrust::sort (random)")
  .add_int64_power_of_two_axis("Elements", nvbench::range(20, 32, 2));
