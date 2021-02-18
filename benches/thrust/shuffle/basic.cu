#include <nvbench/nvbench.cuh>

#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>

template <typename T>
void basic(nvbench::state &state, nvbench::type_list<T>)
{
  const auto size = static_cast<std::size_t>(state.get_int64("Size"));

  thrust::device_vector<T> data(static_cast<std::size_t>(size));
  thrust::sequence(data.begin(), data.end());

  state.add_element_count(size);
  state.add_global_memory_reads<T>(size);
  state.add_global_memory_writes<T>(size);

  auto do_engine = [&state, &data](auto &&engine) {
    using namespace nvbench::exec_tag;
    state.exec(sync, // thrust::shuffle syncs internally
               [&engine, &data](nvbench::launch &launch) {
                 const auto policy = thrust::device.on(launch.get_stream());
                 thrust::shuffle(policy, data.begin(), data.end(), engine);
               });
  };

  const auto rng_engine = state.get_string("Engine");
  if (rng_engine == "minstd")
  {
    do_engine(thrust::random::minstd_rand{});
  }
  else if (rng_engine == "ranlux24")
  {
    do_engine(thrust::random::ranlux24{});
  }
  else if (rng_engine == "ranlux48")
  {
    do_engine(thrust::random::ranlux48{});
  }
  else if (rng_engine == "taus88")
  {
    do_engine(thrust::random::minstd_rand{});
  }
}
using types = nvbench::type_list<nvbench::uint8_t,
                                 nvbench::uint16_t,
                                 nvbench::uint32_t,
                                 nvbench::uint64_t>;
NVBENCH_BENCH_TYPES(basic, NVBENCH_TYPE_AXES(types))
  .set_name("thrust::shuffle")
  .set_type_axes_names({"T"})
  .add_int64_power_of_two_axis("Size", nvbench::range(20, 32, 2))
  .add_string_axis("Engine", {"minstd", "ranlux24", "ranlux48", "taus88"})
  .set_timeout(5);

NVBENCH_MAIN
