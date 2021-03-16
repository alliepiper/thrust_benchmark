/*
 *  Copyright 2021 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 with the LLVM exception
 *  (the "License"); you may not use this file except in compliance with
 *  the License.
 *
 *  You may obtain a copy of the License at
 *
 *      http://llvm.org/foundation/relicensing/LICENSE.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

// This header contains shared code for generating radix_sort input buffers.

#pragma once

#include <nvbench/nvbench.cuh>

// This is not public API because it depends on fmtlib.
// FIXME: Add a similar macro to tbm::
#include <nvbench/detail/throw.cuh>

#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/random.h>

#include <limits>

// stateful host/device random number generator. Generates a uniformly
// distributed number from T's full range of values.
template <typename T>
struct random_value_generator
{
  __host__ __device__ T operator()()
  {
    return static_cast<T>(m_distribution(m_engine));
  }

private:
  static constexpr bool is_float = std::is_floating_point_v<T>;
  using distribution_t =
    std::conditional_t<is_float,
                       thrust::uniform_real_distribution<T>,
                       thrust::uniform_int_distribution<T>>;

  thrust::default_random_engine m_engine{};
  distribution_t m_distribution{std::numeric_limits<T>::lowest(),
                                std::numeric_limits<T>::max()};
};

// Constructs a device vector with a certain type of input. Also handles
// resetting the data to a new random state if needed.
struct input_generator
{
  input_generator(std::size_t size, std::string distribution)
      : m_size(size)
      , m_distribution(std::move(distribution))
  {}

  // Generate a `thrust::vector<T>` with the given configuration.
  template <typename T>
  [[nodiscard]] thrust::device_vector<T> generate() const
  {
    if (m_distribution == "Random")
    {
      return thrust::device_vector<T>(m_size);
    }
    else if (m_distribution == "Constant")
    { // Some non-zero value.
      return thrust::device_vector<T>(m_size, static_cast<T>(1));
    }
    else
    {
      NVBENCH_THROW(std::runtime_error,
                    "Unknown input distribution: {}",
                    m_distribution);
    }
  }

  // Return true if the current configuration requires a reset between
  // executions.
  [[nodiscard]] bool needs_reset() const { return m_distribution == "Random"; }

  // If needs_reset() returns true, call this method to do the actual reset:
  template <typename T>
  void reset(thrust::device_vector<T> &device)
  {
    if (m_distribution == "Random")
    {
      thrust::generate(device.begin(), device.end(), m_engine);
      // Manually advance the host-side engine to a new set of values since the
      // above only increments the engine on the device copy.
      m_engine.discard(m_size);
    }
  }

private:
  std::size_t m_size{};
  std::string m_distribution{};
  thrust::default_random_engine m_engine;
};
