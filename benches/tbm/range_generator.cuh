#pragma once

#include <nvbench/nvbench.cuh>

// This is not public API because it depends on fmtlib.
// FIXME: Add a similar macro to tbm::
#include <nvbench/detail/throw.cuh>

#include <cub/device/device_reduce.cuh>

#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/generate.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/random.h>
#include <thrust/sequence.h>

#include <fmt/format.h>

#include <limits>
#include <optional>
#include <type_traits>

namespace tbm
{

// TODO it would probably be worthwhile to precompile the common instantiations
// of range_generator into a library we can reuse for the benchmark
// executables.

/*! Types of iterators. */
enum class iterator_style
{
  pointer,  ///< A raw device pointer.
  vector,   ///< A `thrust::device_vector` iterator
  counting, ///< A `thrust::counting_iterator`
  constant, ///< A `thrust::constant_iterator`
  discard   ///< A `thrust::discard_iterator`
};

/*! Types of input data pattern. */
enum class data_pattern
{
  none,     ///< Unspecified.
  sequence, ///< A sequence of [0, size).
  constant, ///< A constant array containing values of `T{42}`.
  random    ///< Random values uniformly distributed across `T`'s value range.
};

//==============================================================================
// range_generator_is_valid<T, IteratorStyle, DataPattern = none>
//==============================================================================

// Compile-time check whether an range_generator config is valid:
template <typename T,
          iterator_style IteratorStyle,
          data_pattern DataPattern = data_pattern::none>
struct range_generator_is_valid : std::true_type
{
  // Return true if the state is valid, otherwise set a skip reason on state and
  // return false;.
  static bool should_skip(nvbench::state &) { return false; }
};

// Counting iterators only support sequences
template <typename T, data_pattern DataPattern>
struct range_generator_is_valid<T, iterator_style::counting, DataPattern>
    : std::bool_constant<DataPattern == data_pattern::sequence ||
                         DataPattern == data_pattern::none>
{
  static bool should_skip(nvbench::state &state)
  {
    if constexpr (!value)
    {
      state.skip("Counting iterators only provide sequence patterns.");
    }
    return !value;
  }
};

// Constant iterators only support constant sequences
template <typename T, data_pattern DataPattern>
struct range_generator_is_valid<T, iterator_style::constant, DataPattern>
    : std::bool_constant<DataPattern == data_pattern::constant ||
                         DataPattern == data_pattern::none>
{
  static bool should_skip(nvbench::state &state)
  {
    if constexpr (!value)
    {
      state.skip("Constant iterators only provide constant patterns.");
    }
    return !value;
  }
};

// Discard iterators only support `data_pattern::none`.
template <typename T, data_pattern DataPattern>
struct range_generator_is_valid<T, iterator_style::discard, DataPattern>
    : std::bool_constant<DataPattern == data_pattern::none>
{
  static bool should_skip(nvbench::state &state)
  {
    if constexpr (!value)
    {
      state.skip("Discard iterators only support `data_pattern::none`.");
    }
    return !value;
  }
};

//==============================================================================
// range_generator_needs_reset<T, IteratorStyle, DataPattern = none>
//==============================================================================

// Compile-time check whether an range_generator will need to be reset after
// writing to the iterators.
template <typename T,
          iterator_style IteratorStyle,
          data_pattern DataPattern = data_pattern::none>
struct range_generator_needs_reset : std::true_type
{};

// "none" patterns don't need a reset.
template <typename T, iterator_style IteratorStyle>
struct range_generator_needs_reset<T, IteratorStyle, data_pattern::none>
    : std::false_type
{};

//==============================================================================
// random_value_generator<T>
//==============================================================================

// stateful host/device random number generator. Generates a uniformly
// distributed number from T's full range of values.
template <typename T>
struct random_value_generator
{
  // This functor will call `engine.discard(n)` with large values of n. The
  // default engine is optimized to do this in O(log(n)) time, all others use
  // O(n). Changing this will have a large impact on performance.
  using engine_t = thrust::default_random_engine;

  explicit random_value_generator(engine_t engine)
      : m_engine{engine}
  {}

  template <typename IndexType>
  __host__ __device__ T operator()(IndexType i)
  {
    using discard_type = unsigned long long;

    // Discard multiple values per index in case the distribution uses more than
    // one to generate the result.
    constexpr auto discard_per_i = static_cast<discard_type>(50);
    const auto num_discard = static_cast<discard_type>(i * discard_per_i);
    // This is optimized to O(log(n)) in thrust's default engine:
    m_engine.discard(num_discard);
    return static_cast<T>(m_distribution(m_engine));
  }

private:
  static constexpr bool is_float = std::is_floating_point_v<T>;
  using distribution_t =
    std::conditional_t<is_float,
                       thrust::uniform_real_distribution<T>,
                       thrust::uniform_int_distribution<T>>;

  engine_t m_engine{};
  distribution_t m_distribution{std::numeric_limits<T>::lowest(),
                                std::numeric_limits<T>::max()};
};

//==============================================================================
// range_generator<T, IteratorStyle, DataPattern = none>
//==============================================================================

template <typename T,
          iterator_style IteratorStyle,
          data_pattern DataPattern = data_pattern::none,
          typename                 = void>
struct range_generator;

// Implementation for storage-less iterators
template <typename T, iterator_style IteratorStyle, data_pattern DataPattern>
struct range_generator<
  T,
  IteratorStyle,
  DataPattern,
  typename std::enable_if<IteratorStyle == iterator_style::counting ||
                          IteratorStyle == iterator_style::constant ||
                          IteratorStyle == iterator_style::discard>::type>
{
  // Returns an instance of `range_generator_is_valid<...>`, which inherits
  // from `std::bool_constant` and provides a bit of extra API for skipping
  // invalid benchmarks.
  [[nodiscard]] static constexpr auto is_valid()
  {
    return range_generator_is_valid<T, IteratorStyle, DataPattern>{};
  }

  // No state, nothing to reset.
  [[nodiscard]] static constexpr auto needs_reset()
  {
    return std::false_type{};
  }

  void init(std::size_t size) { m_size = size; }

  [[nodiscard]] bool should_skip(nvbench::state &state) const
  {
    if constexpr (IteratorStyle == iterator_style::counting)
    {
      // Counting iterators store the range size in a `T`. We'll need to skip
      // types that can't hold m_size.
      if (m_size > std::numeric_limits<T>::max())
      {
        state.skip(fmt::format("thrust::counting_iterator<{}> max range "
                               "exceeded ({} > {}).",
                               nvbench::demangle<T>(),
                               m_size,
                               std::numeric_limits<T>::max()));
        return true;
      }
    }
    return this->is_valid().should_skip(state);
  }

  [[nodiscard]] std::size_t get_allocation_size() { return 0; }

  void reset() {} // no storage, no op

  [[nodiscard]] auto begin() { return this->cbegin(); }
  [[nodiscard]] auto end() { return this->cend(); }
  [[nodiscard]] auto begin() const { return this->cbegin(); }
  [[nodiscard]] auto end() const { return this->cend(); }

  [[nodiscard]] auto cbegin() const
  {
    if constexpr (IteratorStyle == iterator_style::counting)
    {
      return thrust::make_counting_iterator<T>(T{});
    }
    else if constexpr (IteratorStyle == iterator_style::constant)
    {
      // FIXME: Using 64-bit indices probably isn't ideal here, since we have
      // the option to use different `IndexT` with constant_iterator.
      return thrust::make_constant_iterator<T>(static_cast<T>(42),
                                               std::size_t{});
    }
    else if constexpr (IteratorStyle == iterator_style::discard)
    {
      return thrust::make_discard_iterator(std::size_t{});
    }
    NVBENCH_THROW(std::runtime_error, "{}", "Unknown iterator_style");
  }

  [[nodiscard]] auto cend() const
  {
    if constexpr (IteratorStyle == iterator_style::counting)
    {
      return thrust::make_counting_iterator<T>(static_cast<T>(m_size));
    }
    else if constexpr (IteratorStyle == iterator_style::constant)
    {
      // FIXME: Using 64-bit indices probably isn't ideal here, since we have
      // the option to use different `IndexT` with constant_iterator.
      return thrust::make_constant_iterator<T>(static_cast<T>(42), m_size);
    }
    else if constexpr (IteratorStyle == iterator_style::discard)
    {
      return thrust::make_discard_iterator(m_size);
    }
    NVBENCH_THROW(std::runtime_error, "{}", "Unknown iterator_style");
  }

private:
  std::size_t m_size{};
};

// Implementation for global-memory backed iterators
template <typename T, iterator_style IteratorStyle, data_pattern DataPattern>
struct range_generator<
  T,
  IteratorStyle,
  DataPattern,
  typename std::enable_if<IteratorStyle == iterator_style::pointer ||
                          IteratorStyle == iterator_style::vector>::type>
{
  // Returns an instance of `range_generator_is_valid<...>`, which inherits
  // from `std::bool_constant` and provides a bit of extra API for skipping
  // invalid benchmarks.
  [[nodiscard]] static constexpr auto is_valid()
  {
    return range_generator_is_valid<T, IteratorStyle, DataPattern>{};
  }

  // Returns `std::bool_constant`
  [[nodiscard]] static constexpr auto needs_reset()
  {
    return range_generator_needs_reset<T, IteratorStyle, DataPattern>{};
  }

  void init(std::size_t size)
  {
    m_data.resize(size);
    if constexpr (DataPattern == data_pattern::random)
    {
      m_engine = thrust::default_random_engine{};
    }
    this->reset();
  }

  [[nodiscard]] bool should_skip(nvbench::state &state) const
  {
    return this->is_valid().should_skip(state);
  }

  [[nodiscard]] std::size_t get_allocation_size() const
  {
    return sizeof(T) * m_data.size();
  }

  void reset()
  {
    if constexpr (needs_reset())
    {
      if constexpr (DataPattern == data_pattern::sequence)
      {
        thrust::sequence(m_data.begin(), m_data.end());
      }
      else if constexpr (DataPattern == data_pattern::constant)
      {
        thrust::fill(m_data.begin(), m_data.end(), static_cast<T>(42));
      }
      else if constexpr (DataPattern == data_pattern::random)
      {
        auto &engine = m_engine.value();
        thrust::tabulate(m_data.begin(),
                         m_data.end(),
                         random_value_generator<T>{engine});

        // Manually advance the host-side engine to a new set of values since
        // the above only increments the engine on the device copy.
        engine.discard(m_data.size());
      }
    }
  }

  [[nodiscard]] auto begin()
  {
    if constexpr (IteratorStyle == iterator_style::pointer)
    {
      return thrust::raw_pointer_cast(m_data.data());
    }
    else if constexpr (IteratorStyle == iterator_style::vector)
    {
      return m_data.begin();
    }
    NVBENCH_THROW(std::runtime_error, "{}", "Unknown iterator_style");
  }

  [[nodiscard]] auto end()
  {
    if constexpr (IteratorStyle == iterator_style::pointer)
    {
      return thrust::raw_pointer_cast(m_data.data()) + m_data.size();
    }
    else if constexpr (IteratorStyle == iterator_style::vector)
    {
      return m_data.end();
    }
    NVBENCH_THROW(std::runtime_error, "{}", "Unknown iterator_style");
  }

  [[nodiscard]] auto begin() const { return this->cbegin(); }
  [[nodiscard]] auto end() const { return this->cend(); }

  [[nodiscard]] auto cbegin() const
  {
    if constexpr (IteratorStyle == iterator_style::pointer)
    {
      return thrust::raw_pointer_cast(m_data.data());
    }
    else if constexpr (IteratorStyle == iterator_style::vector)
    {
      return m_data.cbegin();
    }
    NVBENCH_THROW(std::runtime_error, "{}", "Unknown iterator_style");
  }

  [[nodiscard]] auto cend() const
  {
    if constexpr (IteratorStyle == iterator_style::pointer)
    {
      return thrust::raw_pointer_cast(m_data.data()) + m_data.size();
    }
    else if constexpr (IteratorStyle == iterator_style::vector)
    {
      return m_data.cend();
    }
    NVBENCH_THROW(std::runtime_error, "{}", "Unknown iterator_style");
  }

private:
  thrust::device_vector<T> m_data;
  std::optional<thrust::default_random_engine> m_engine;
};

template <typename T,
          iterator_style IteratorStyle = iterator_style::pointer,
          data_pattern DataPattern     = data_pattern::none>
[[nodiscard]] auto make_range_generator(std::size_t size)
{
  using result_t = tbm::range_generator<T, IteratorStyle, DataPattern>;
  result_t result;
  result.init(size);
  return result;
}

} // namespace tbm

// These must be in the global namespace:
NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  tbm::iterator_style,
  [](tbm::iterator_style style) {
    switch (style)
    {
      case tbm::iterator_style::pointer:
        return "Pointer";
      case tbm::iterator_style::vector:
        return "Vector";
      case tbm::iterator_style::counting:
        return "Counting";
      case tbm::iterator_style::constant:
        return "Constant";
      case tbm::iterator_style::discard:
        return "Discard";
      default:
        break;
    }
    NVBENCH_THROW(std::runtime_error, "{}", "Unknown iterator_style");
  },
  [](tbm::iterator_style style) {
    switch (style)
    {
      case tbm::iterator_style::pointer:
        return "`T*`";
      case tbm::iterator_style::vector:
        return "`thrust::device_vector::iterator`";
      case tbm::iterator_style::counting:
        return "`thrust::counting_iterator`";
      case tbm::iterator_style::constant:
        return "`thrust::constant_iterator`";
      case tbm::iterator_style::discard:
        return "`thrust::discard_iterator`";
      default:
        break;
    }
    NVBENCH_THROW(std::runtime_error, "{}", "Unknown iterator_style");
  })

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  tbm::data_pattern,
  [](tbm::data_pattern pattern) {
    switch (pattern)
    {
      case tbm::data_pattern::none:
        return "None";
      case tbm::data_pattern::sequence:
        return "Seq";
      case tbm::data_pattern::constant:
        return "Const";
      case tbm::data_pattern::random:
        return "Rand";
      default:
        break;
    }
    NVBENCH_THROW(std::runtime_error, "{}", "Unknown data_pattern");
  },
  [](tbm::data_pattern style) {
    switch (style)
    {
      case tbm::data_pattern::none:
        return "Unspecified.";
      case tbm::data_pattern::sequence:
        return "Sequence of [0, size)";
      case tbm::data_pattern::constant:
        return "All values = 42";
      case tbm::data_pattern::random:
        return "Random values uniformly distributed across `T`'s value range";
      default:
        break;
    }
    NVBENCH_THROW(std::runtime_error, "{}", "Unknown data_pattern");
  })
