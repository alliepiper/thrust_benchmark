#include <cub/device/device_reduce.cuh>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>

#include <nvbench/nvbench.cuh>

#include <type_traits>

template <typename AgentPolicy>
struct FauxDispatchPolicy
{
  struct MaxPolicy : cub::ChainedPolicy<0, MaxPolicy, MaxPolicy>
  {
    using ReducePolicy     = AgentPolicy;
    using SingleTilePolicy = AgentPolicy;
  };
};

template <typename T, T Val>
using static_val = std::integral_constant<T, Val>;

template <typename T, T... Vals>
using static_vals = nvbench::type_list<static_val<T, Vals>...>;

using block_thread_axis    = static_vals<int, 128, 256, 512>;
using items_per_block_axis = static_vals<int, 12, 16, 20, 24>;
using compute_t_axis       = nvbench::type_list<nvbench::int32_t>; // for now...
using vector_load_length_axis = static_vals<int, 2, 4, 6>;
using block_algorithm_axis    = static_vals<cub::BlockReduceAlgorithm,
  cub::BLOCK_REDUCE_RAKING,
  cub::BLOCK_REDUCE_WARP_REDUCTIONS>;
using cache_load_modifier_axis =
static_vals<cub::CacheLoadModifier, cub::LOAD_DEFAULT>;

template <int BlockThreads4B,
  int ItemsPerBlock4B,
  typename ComputeT,
  int VectorLoadLength,
  cub::BlockReduceAlgorithm BlockAlgorithm,
  cub::CacheLoadModifier LoadModifier>
void tune_cub_reduce(
  nvbench::state &state,
  nvbench::type_list<static_val<int, BlockThreads4B>,
    static_val<int, ItemsPerBlock4B>,
    typename ComputeT,
    static_val<int, VectorLoadLength>,
    static_val<cub::BlockReduceAlgorithm, BlockAlgorithm>,
    static_val<cub::CacheLoadModifier, LoadModifier>>)
{
  using AgentPolicy = cub::AgentReducePolicy<BlockThreads4B,
    ItemsPerBlock4B,
    ComputeT,
    VectorLoadLength,
    BlockAlgorithm,
    LoadModifier>;

  // We need a more direct way of invoking the algorithm with a custom policy...
  // I'll add one as part of tuning, power users will love it.
  using DispatchPolicy = FauxDispatchPolicy<AgentPolicy>;

  const auto elements = static_cast<std::size_t>(state.get_int64("Elements"));

  state.add_element_count(elements);
  state.add_global_memory_reads<ComputeT>(elements);
  state.add_global_memory_writes<ComputeT>(1);

  thrust::device_vector<ComputeT> data(elements + 1); // output at end
  thrust::sequence(data.begin(), data.end());

  using input_iterator_t  = std::decay_t<decltype(data.cbegin())>;
  using output_iterator_t = std::decay_t<decltype(data.begin())>;

  using Dispatcher = cub::DispatchReduce<input_iterator_t,
    output_iterator_t,
    /*OffsetT*/ int,
    cub::Sum,
    ComputeT,
    DispatchPolicy>;

  std::size_t temp_size{};
  Dispatcher::Dispatch(nullptr,
                       temp_size,
                       data.cbegin(),
                       data.begin() + elements,
                       elements,
                       cub::Sum{},
                       ComputeT{},
                       0,
                       false);
  thrust::device_vector<nvbench::uint8_t> temp(temp_size);

  state.exec([&data, &temp, &temp_size, elements](nvbench::launch &launch) {
    Dispatcher::Dispatch(thrust::raw_pointer_cast(temp.data()),
                         temp_size,
                         data.cbegin(),
                         data.begin() + elements,
                         elements,
                         cub::Sum{},
                         ComputeT{},
                         launch.get_stream(),
                         false);
  });
}
NVBENCH_BENCH_TYPES(tune_cub_reduce,
                    NVBENCH_TYPE_AXES(block_thread_axis,
                                      items_per_block_axis,
                                      compute_t_axis,
                                      vector_load_length_axis,
                                      block_algorithm_axis,
                                      cache_load_modifier_axis))
.set_name("cub::DeviceReduce tuning experiment")
  .set_type_axes_names(
    {"th/bl", "elem/bl", "T", "vec_ld", "blk_algo", "load_mod"})
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 32, 2))
  .set_timeout(1);

NVBENCH_MAIN
