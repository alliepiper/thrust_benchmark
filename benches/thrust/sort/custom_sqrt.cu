#include <nvbench/nvbench.cuh>

#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>
#include <thrust/sort.h>

/*
 * Below you can find a type that provides an arbitrary amount of operations.
 * By changing the number of operations, you can control the comparison cost.
 */

template <typename data_type, unsigned int operations_left>
struct helper;

template <typename data_type>
struct helper<data_type, 1>
{
  static __device__ bool compute(data_type lhs, data_type rhs)
  {
    return lhs < rhs;
  }
};

template <typename data_type, unsigned int operations_left>
struct helper
{
  static __device__ bool compute(data_type lhs, data_type rhs)
  {
    return helper<data_type, operations_left - 1>::compute(sqrt(lhs * lhs),
                                                           sqrt(rhs * rhs));
  }
};

template <typename data_type, unsigned int operations_count>
struct comparator
{
  __device__ bool operator()(data_type lhs, data_type rhs)
  {
    return helper<data_type, operations_count>::compute(lhs, rhs);
  }
};

template <typename data_type, int operations_count>
void custom_sqrt(nvbench::state &state,
                 nvbench::type_list<data_type,
                                    nvbench::enum_type<operations_count>>)
{
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements"));

  constexpr data_type small_value = 1e-7;
  thrust::device_vector<data_type> data(elements);
  thrust::sequence(data.begin(), data.end(), small_value, small_value);

  thrust::default_random_engine rng;

  state.add_element_count(elements);
  state.add_global_memory_reads<data_type>(elements);
  state.add_global_memory_writes<data_type>(elements);

  using namespace nvbench::exec_tag;
  state.exec(timer | sync, // This benchmark needs a timer and syncs internally
             [&rng, &data](nvbench::launch &launch, auto &timer) {
               const auto policy = thrust::device.on(launch.get_stream());
               thrust::shuffle(policy, data.begin(), data.end(), rng);
               timer.start();
               thrust::sort(policy,
                            data.begin(),
                            data.end(),
                            comparator<data_type, operations_count>{});
               timer.stop();
             });
}
using types = nvbench::type_list<nvbench::float32_t>;
using ops = nvbench::enum_type_list<1,2,4,8,16>;
NVBENCH_BENCH_TYPES(custom_sqrt, NVBENCH_TYPE_AXES(types, ops))
  .set_name("thrust::sort<custom_sqrt> (random)")
  .add_int64_power_of_two_axis("Elements", nvbench::range(20, 30, 2));
