# `cub::DeviceReduce` Notes

cub::DeviceReduce offers several convenience entry points: sum, min, max, etc.
Most of these follow a generic path.

Sum, however, is specialized for the following types (see `WarpReduceShfl`):

- float
- unsigned int
- unsigned long long
- long long
- double

Calling `cub::DeviceReduce::Sum(...)` vs
`cub::DeviceReduce::Reduce(..., cub::Sum{})` makes no difference. `Sum` just
calls `Reduce(..., Sum{})` internally.
