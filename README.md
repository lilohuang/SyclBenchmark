# SyclBenchmark

SyclBenchmark is a C++17 command-line tool for measuring SYCL device
throughput and running bounded compute/VRAM stability tests. Its kernels do not
call cuBLAS, CUTLASS, oneMKL, or another compute library.

## Quick start

```sh
# Detect locally available backends and build one binary
make BACKEND=auto

# Find stable backend:index selectors
./sycl-bench devices

# Run the complete benchmark on one device
./sycl-bench benchmark --device cuda:0

# Run a 30-minute mixed stress test
./sycl-bench stress --device cuda:0 --duration 30m --profile mixed
```

Running `sycl-bench` without a command prints help. Benchmark and stress
commands require an explicit device selection so an accidental invocation
cannot load every visible device.

## Capabilities

The benchmark command measures:

- FP32 and, when supported, FP64 fused multiply-add throughput
- FP16, TF32, INT8, and FP64 joint-matrix throughput
- Device-memory read and write bandwidth
- Host-to-device, device-to-host, and concurrent USM transfer bandwidth

The stress command provides:

- `compute`, `vram`, and `mixed` profiles
- Selectable FP32, FP64, FP16/TF32/INT8/FP64 joint-matrix compute workloads
- Full compute-output validation after every batch
- Full working-set validation after each address-dependent VRAM pattern write
- Bounded durations, reproducible seeds, and graceful Ctrl-C handling
- Parallel multi-GPU execution by default, with a sequential option
- Interval reports plus final average, minimum, and P95 statistics

## Requirements

- A C++17 SYCL compiler and runtime
- Device USM support for benchmark and stress kernels
- Host USM support for transfer measurements
- CUDA and a compatible NVIDIA driver for CUDA builds
- ROCm and a compatible AMD GPU for HIP builds
- A 32-lane subgroup and matching `matrix_combinations` entry for matrix tests

To reproduce the architecture-specific matrix results, use the custom
[lilohuang/llvm commit `01516f0d8a96b135c3ce45405adbaf2f3a7cd336`](https://github.com/lilohuang/llvm/commit/01516f0d8a96b135c3ce45405adbaf2f3a7cd336).
Other SYCL compiler revisions may build the program but are not expected to
provide the same matrix coverage or results.

The default compiler location is:

```text
~/sycl_workspace/llvm/build/bin/clang++
```

Override it with `SYCL_ROOT=/path/to/llvm/build`.

## Build

`BACKEND=auto` enables generic SPIR-V plus locally detected CUDA and HIP
targets. CUDA and AMD architectures are detected from `nvidia-smi` and
`rocm_agent_enumerator`.

```sh
make BACKEND=auto
make config
```

Explicit builds:

```sh
make BACKEND=cuda
make BACKEND=hip
make BACKEND=spirv
make BACKEND=all

# Cross-build or override architecture detection
make BACKEND=cuda CUDA_ARCH=sm_86
make BACKEND=hip AMD_GPU_ARCH=gfx1102
```

Convenience targets are equivalent to setting `BACKEND`:

```sh
make cuda
make hip
make spirv
make all
```

AMD's signed-character libspirv compatibility alias is created inside a
configuration-specific `build/clang-resource-*` directory; the build no longer
modifies the compiler installation.
Override the bitcode location when necessary:

```sh
make BACKEND=hip AMD_LIBSPIRV=/path/to/libspirv.l64.signed_char.bc
```

## Device discovery

```sh
./sycl-bench devices
./sycl-bench devices --format json
./sycl-bench devices --format csv --output devices.csv
```

The text table includes the global ID, stable `backend:index` selector, device
type, reported memory, device-USM availability, and matrix capability. Prefer a
selector such as `cuda:0`, `hip:0`, or `level_zero:0`; global numeric IDs can
change when runtimes are added or removed.

`--device` also accepts unique case-insensitive text from a device name:

```sh
./sycl-bench benchmark --device "Radeon Pro W7500"
```

Ambiguous names are rejected with a request to use `backend:index`.

## Benchmark

Run every test on one or more devices:

```sh
./sycl-bench benchmark --device cuda:0
./sycl-bench benchmark --device cuda:0 --device hip:0
./sycl-bench benchmark --all
```

Benchmark selections run sequentially so measurements from different devices
do not compete for host or link bandwidth.

Select a subset with a comma-separated list:

```sh
./sycl-bench benchmark --device hip:0 --tests fp32,memory
./sycl-bench benchmark --device cuda:0 \
  --tests matrix-fp16,matrix-tf32,matrix-int8
```

Available tests are:

```text
fp64, fp32, matrix-fp16, matrix-tf32, matrix-int8,
matrix-fp64, memory, transfer
```

The benchmark warms up each test and reports the best of multiple runs. Kernel
event profiling is used when available, with host timing as a fallback. FMA is
counted as two operations. Unsupported tests are reported as `unavailable`.

Matrix output contains complete GEMM throughput and an `.issue` measurement.
The latter is an instruction issue-rate ceiling, not complete GEMM throughput.

All matrix tests measure dense MMA throughput through the ordinary
`joint_matrix_mad` path. Compare a reported rate only with a dense MMA peak
that uses the same input and accumulator types.

These tests do not issue sparse MMA instructions, encode structured-sparsity
metadata, or compress sparse operands. A theoretical peak that assumes
sparsity or reports an effective sparse rate is therefore not directly
comparable and is not an expected result. Zero-valued matrix elements are
still processed and counted as part of a dense operation.

## Stress test

A duration and explicit device selection are required:

```sh
./sycl-bench stress --device hip:0 --duration 30m
./sycl-bench stress --device cuda:0 --duration 1h --profile compute
./sycl-bench stress --device cuda:0 --duration 1h --profile compute \
  --compute-workload fp64
./sycl-bench stress --device level_zero:0 --duration 30m --profile compute \
  --compute-workload matrix-fp16
./sycl-bench stress --device hip:0 --duration 8h \
  --profile vram --memory 80
```

Durations accept `s`, `m`, and `h`. The defaults are:

```text
profile=mixed  memory=50%  chunk-size=512MiB  report-interval=5s
seed=0x6d2b79f5  execution=parallel
```

The default compute workload is `fp32`. `--compute-workload` also accepts
`fp64`, `matrix-fp16`, `matrix-tf32`, `matrix-int8`, and `matrix-fp64` for the
`compute` and `mixed` profiles. A requested FP64 or joint-matrix workload is
never replaced with FP32: if the selected device does not advertise the
required FP64 aspect, 32-lane subgroup, or matching `matrix_combinations`
entry, that device reports `error` and the command exits with status 1. Matrix
workloads use the same dense matrix path described above and perform a full
output verification after every timed batch.

Multiple selected devices run concurrently for the requested wall-clock
duration:

```sh
./sycl-bench stress --device cuda:0 --device hip:0 --duration 30m
./sycl-bench stress --all-gpus --duration 30m
```

When OpenCL and Level Zero expose the same PCI address or device UUID,
`--all-gpus` keeps the Level Zero view so that physical GPU is not stressed
twice. Separate GPUs of the same model remain selected. Explicit `--device`
selections are never deduplicated.

Use `--sequential` when board power or cooling cannot support simultaneous
loads. In that mode, each selected device receives the full duration.

The memory percentage is limited to 1-90. Integrated devices often report most
of system RAM as global memory, so their percentage is based on a conservative
8 GiB ceiling. The startup plan shows the requested percentage and the final
summary shows the actual tested MiB.

Every device uses chunked VRAM allocation. The default maximum is 512 MiB and
can be changed with `--chunk-size`; a bare number means MiB, and `MiB` or `GiB`
suffixes are accepted. The actual chunk size is also capped by the device's
reported single-allocation limit. The last chunk carries the exact remainder,
so the combined allocation still matches `--memory`. Pattern writes and full
verification run across every chunk.

```sh
./sycl-bench stress --device level_zero:0 --duration 30m \
  --memory 50 --chunk-size 256MiB
```

Reproduce a VRAM pattern sequence with:

```sh
./sycl-bench stress --device hip:0 --duration 10m --seed 0x12345678
```

Automation can turn performance regressions into failures:

```sh
./sycl-bench stress --device hip:0 --duration 30m \
  --min-compute-rate 8.0 --min-vram-rate 50 \
  --max-slowdown 20
```

Minimum rates use the final full-run average. Slowdown compares the first
reporting interval with the slowest later interval. A violated threshold exits
with status 1 and is included in text and structured summaries. Mixed stress
alternates compute and VRAM work and requires at least one valid sample from
each before it can report a pass.

Ctrl-C performs an orderly stop after the active kernel completes.

## Structured output

All commands support `text`, `json`, `jsonl`, and `csv`:

```sh
./sycl-bench benchmark --device cuda:0 --format json \
  --output benchmark.json
./sycl-bench stress --device hip:0 --duration 8h --format jsonl \
  --output stress.jsonl
```

`json` produces one final document. During stress tests, `jsonl` and `csv`
flush every interval and are better choices for durable long-running logs.
Use `--output -` or omit `--output` for standard output.

## Exit status

```text
0    completed successfully
1    benchmark or stress workload failed
2    invalid command, configuration, or device selection
130  interrupted with Ctrl-C or SIGTERM
```

## Measurement notes

- `concurrent-both` is the aggregate bandwidth of both transfer directions.
- Device memory capacity can represent shared system memory on integrated GPUs.
- SYCL's maximum clock property is static metadata, not a live clock reading.
- Temperature, power, fan speed, and live clocks are not portable SYCL
  properties; monitor them with the hardware vendor's system tools.
- Results vary with compiler, backend, clocks, power state, and memory path.

## License

Released under the [MIT License](LICENSE).

Copyright (c) 2026 Lilo Huang.
