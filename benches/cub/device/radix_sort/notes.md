# `cub::DeviceRadixSort` Notes

## Shared headers:

- `*_bench.cuh`: Benchmark implementations.
- `type_lists.cuh`: Predefined type axis helpers.

## Layout

- SortDirection and DataPattern are passed as compile-time parameters to avoid
  instantiating unused templates.
- Descending benchmarks are kept in a separate TU because they instantiate
  different kernels than the ascending versions. This allows them to compile in
  parallel.
