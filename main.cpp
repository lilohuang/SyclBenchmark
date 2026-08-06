// Copyright (c) 2026, Lilo Huang <kuso.cc@gmail.com>
// SPDX-License-Identifier: MIT

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/info/device.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cctype>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <sstream>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
namespace matrix = sycl::ext::oneapi::experimental::matrix;
constexpr const char *tool_version = "1.1.0";

std::string concise_error(const std::exception &error) {
   const auto *sycl_error = dynamic_cast<const sycl::exception *>(&error);
   if ((sycl_error &&
          sycl_error->code() ==
             sycl::make_error_code(sycl::errc::kernel_not_supported)) ||
       std::string(error.what()).find(
          "has no image for the selected device") != std::string::npos) {
      return "no compatible device image";
   }
   return error.what();
}

enum class OutputFormat { text, json, jsonl, csv };

const char *backend_name(sycl::backend backend) {
   switch (backend) {
   case sycl::backend::opencl:
      return "opencl";
   case sycl::backend::ext_oneapi_level_zero:
      return "level_zero";
   case sycl::backend::ext_oneapi_cuda:
      return "cuda";
   case sycl::backend::ext_oneapi_hip:
      return "hip";
   case sycl::backend::ext_oneapi_native_cpu:
      return "native_cpu";
   case sycl::backend::ext_oneapi_offload:
      return "offload";
   default:
      return "unknown";
   }
}

const char *device_type_name(const sycl::device &device) {
   if (device.is_gpu()) {
      return "gpu";
   }
   if (device.is_cpu()) {
      return "cpu";
   }
   if (device.is_accelerator()) {
      return "accelerator";
   }
   return "custom";
}

struct DeviceEntry {
   std::size_t id;
   std::size_t backend_id;
   std::string selector;
   sycl::device device;
};

std::vector<DeviceEntry> discover_devices() {
   const auto devices = sycl::device::get_devices();
   std::vector<DeviceEntry> result;
   std::vector<std::pair<sycl::backend, std::size_t>> counts;
   for (std::size_t id = 0; id < devices.size(); ++id) {
      const auto backend = devices[id].get_backend();
      auto found = std::find_if(counts.begin(), counts.end(),
         [&](const auto &item) { return item.first == backend; });
      if (found == counts.end()) {
         counts.push_back({backend, 0});
         found = std::prev(counts.end());
      }
      const std::size_t backend_id = found->second++;
      result.push_back({id, backend_id,
         std::string(backend_name(backend)) + ':' +
            std::to_string(backend_id), devices[id]});
   }
   return result;
}

std::string json_escape(const std::string &value) {
   std::ostringstream out;
   for (const unsigned char c : value) {
      switch (c) {
      case '"':
         out << "\\\"";
         break;
      case '\\':
         out << "\\\\";
         break;
      case '\b':
         out << "\\b";
         break;
      case '\f':
         out << "\\f";
         break;
      case '\n':
         out << "\\n";
         break;
      case '\r':
         out << "\\r";
         break;
      case '\t':
         out << "\\t";
         break;
      default:
         if (c < 0x20) {
            out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                << static_cast<unsigned>(c) << std::dec << std::setfill(' ');
         } else {
            out << static_cast<char>(c);
         }
      }
   }
   return out.str();
}

std::string csv_escape(const std::string &value) {
   if (value.find_first_of(",\"\r\n") == std::string::npos) {
      return value;
   }
   std::string escaped = "\"";
   for (const char c : value) {
      escaped += c == '"' ? "\"\"" : std::string(1, c);
   }
   return escaped + '"';
}

std::string utc_timestamp() {
   const std::time_t now = std::time(nullptr);
   std::tm utc{};
   gmtime_r(&now, &utc);
   std::ostringstream out;
   out << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
   return out.str();
}

template <typename F> double elapsed_ms(F &&work) {
   const auto begin = Clock::now();
   work();
   return std::chrono::duration<double, std::milli>(Clock::now() - begin)
      .count();
}

template <typename F> double best_ms(F &&work, int runs = 3) {
   work();
   double best = std::numeric_limits<double>::max();
   for (int i = 0; i < runs; ++i) {
      best = std::min(best, elapsed_ms(work));
   }
   return best;
}

sycl::queue timing_queue(const sycl::context &context,
   const sycl::device &device, bool profiling) {
   if (profiling) {
      return sycl::queue(context, device,
         sycl::property_list{sycl::property::queue::in_order{},
                             sycl::property::queue::enable_profiling{}});
   }
   return sycl::queue(context, device,
      sycl::property_list{sycl::property::queue::in_order{}});
}

template <typename F>
double best_kernel_ns(F &&launch, bool profiling, int runs = 3) {
   launch().wait_and_throw();
   double best = std::numeric_limits<double>::max();
   for (int i = 0; i < runs; ++i) {
      if (profiling) {
         auto event = launch();
         event.wait_and_throw();
         const auto begin = event.template get_profiling_info<
            sycl::info::event_profiling::command_start>();
         const auto end = event.template get_profiling_info<
            sycl::info::event_profiling::command_end>();
         if (end <= begin) {
            throw std::runtime_error("invalid SYCL profiling timestamps");
         }
         best = std::min(best, static_cast<double>(end - begin));
      } else {
         best = std::min(best,
            elapsed_ms([&] { launch().wait_and_throw(); }) * 1.0e6);
      }
   }
   return best;
}

template <typename T> class UsmBuffer {
public:
   UsmBuffer(sycl::queue &q, std::size_t count, bool host = false)
      : q_(q), ptr_(host ? sycl::malloc_host<T>(count, q)
                         : sycl::malloc_device<T>(count, q)) {
      if (!ptr_) {
         throw std::bad_alloc();
      }
   }

   ~UsmBuffer() { sycl::free(ptr_, q_); }
   UsmBuffer(const UsmBuffer &) = delete;
   UsmBuffer &operator=(const UsmBuffer &) = delete;
   T *get() const { return ptr_; }

private:
   sycl::queue &q_;
   T *ptr_;
};

double gbps(std::size_t bytes, double ms) {
   return static_cast<double>(bytes) / (ms * 1.0e6);
}

template <typename T>
double compute(sycl::queue &q, const sycl::device &device) {
   constexpr std::size_t fma_accumulators = 16;
   const std::size_t wg = std::min<std::size_t>(256,
      device.get_info<sycl::info::device::max_work_group_size>());
   const std::size_t groups = std::max<std::size_t>(256,
      8 * device.get_info<sycl::info::device::max_compute_units>());
   const std::size_t items = wg * groups;
   const bool profiling = device.has(sycl::aspect::queue_profiling);
   sycl::queue cq = timing_queue(q.get_context(), device, profiling);
   UsmBuffer<T> output(cq, items);
   T *out = output.get();

   cq.parallel_for(sycl::range<1>(items),
      [=](sycl::id<1> i) { out[i] = T(1); })
      .wait_and_throw();

   auto launch = [&](std::size_t iterations) {
      return cq.parallel_for(sycl::range<1>(items), [=](sycl::id<1> id) {
         T x = out[id];
         T acc[fma_accumulators];
#pragma unroll
         for (std::size_t lane = 0; lane < fma_accumulators; ++lane) {
            acc[lane] = x + T(lane + 1);
         }
         const T mul = T(0.999f), add = T(0.001f);
         for (std::size_t i = 0; i < iterations; ++i) {
#pragma unroll
            for (std::size_t lane = 0; lane < fma_accumulators; ++lane) {
               acc[lane] = sycl::fma(acc[lane], mul, add);
            }
         }
         T sum{};
#pragma unroll
         for (std::size_t lane = 0; lane < fma_accumulators; ++lane) {
            sum += acc[lane];
         }
         out[id] = sum;
      });
   };

   std::size_t iterations = 256;
   for (int attempt = 0; attempt < 4; ++attempt) {
      const double sample =
         elapsed_ms([&] { launch(iterations).wait_and_throw(); });
      if (sample >= 30.0 || iterations == (1u << 20)) {
         break;
      }
      const auto scale = static_cast<std::size_t>(
         std::clamp(80.0 / std::max(sample, 0.05), 2.0, 32.0));
      iterations = std::min<std::size_t>(1u << 20, iterations * scale);
   }

   const double ns =
      best_kernel_ns([&] { return launch(iterations); }, profiling);
   T sink{};
   cq.memcpy(&sink, out, sizeof(T)).wait_and_throw();
   if (!std::isfinite(static_cast<double>(sink)) || sink <= T{}) {
      throw std::runtime_error("arithmetic verification failed");
   }
   const double operations = static_cast<double>(items) * iterations *
      fma_accumulators * 2;
   return operations / ns / 1.0e3;
}

struct BenchmarkMeasurement {
   std::string category;
   std::string name;
   std::string status = "ok";
   double value = 0.0;
   std::string unit;
   std::string detail;
   std::string message;
};

template <typename T>
BenchmarkMeasurement benchmark_compute(sycl::queue &q,
   const sycl::device &device, const char *name) {
   BenchmarkMeasurement result{"arithmetic", name};
   try {
      result.value = compute<T>(q, device);
      result.unit = "TFLOP/s";
   } catch (const std::exception &e) {
      result.message = concise_error(e);
      result.status = result.message.find("verification failed") !=
            std::string::npos
         ? "error" : "unavailable";
   }
   return result;
}

struct MatrixRate {
   std::size_t size;
   double tops;
};

enum MatrixImage : unsigned {
   matrix_image_spirv = 1u << 0,
   matrix_image_cuda_sm70 = 1u << 1,
   matrix_image_cuda_sm72 = 1u << 2,
   matrix_image_cuda_sm80 = 1u << 3,
   matrix_image_amd_gfx11 = 1u << 4,
   matrix_image_amd_gfx90a = 1u << 5
};

template <typename Storage_, typename Element_, typename Acc_,
   std::size_t M_, std::size_t N_, std::size_t K_, std::size_t Subgroup_,
   unsigned Images_, matrix::matrix_type InputType_,
   matrix::matrix_type AccType_>
struct MatrixCase {
   using storage_type = Storage_;
   using element_type = Element_;
   using acc_type = Acc_;
   static constexpr std::size_t m = M_;
   static constexpr std::size_t n = N_;
   static constexpr std::size_t k = K_;
   static constexpr std::size_t subgroup = Subgroup_;
   static constexpr unsigned images = Images_;
   static constexpr std::size_t rows = 4;
   static constexpr std::size_t cols = 4;
   static constexpr matrix::matrix_type input_type = InputType_;
   static constexpr matrix::matrix_type acc_matrix_type = AccType_;
};

constexpr unsigned cuda_fp16_images = matrix_image_cuda_sm70 |
   matrix_image_cuda_sm72 | matrix_image_cuda_sm80;
constexpr unsigned cuda_int8_images =
   matrix_image_cuda_sm72 | matrix_image_cuda_sm80;

using Fp16Case = MatrixCase<sycl::half, sycl::half, float, 16, 16, 16, 32,
   cuda_fp16_images | matrix_image_amd_gfx11,
   matrix::matrix_type::fp16, matrix::matrix_type::fp32>;
using Bf16Case = MatrixCase<sycl::ext::oneapi::bfloat16,
   sycl::ext::oneapi::bfloat16, float, 16, 16, 16, 32,
   matrix_image_cuda_sm80 | matrix_image_amd_gfx11,
   matrix::matrix_type::bf16, matrix::matrix_type::fp32>;
using Tf32Case = MatrixCase<float, matrix::precision::tf32, float, 16, 16, 8,
   32, matrix_image_cuda_sm80,
   matrix::matrix_type::tf32, matrix::matrix_type::fp32>;
using Int8Case = MatrixCase<std::int8_t, std::int8_t, std::int32_t, 16, 16,
   16, 32, cuda_int8_images | matrix_image_amd_gfx11,
   matrix::matrix_type::sint8, matrix::matrix_type::sint32>;
using Fp64Case = MatrixCase<double, double, double, 8, 8, 4, 32,
   matrix_image_cuda_sm80,
   matrix::matrix_type::fp64, matrix::matrix_type::fp64>;

using Amd90aFp16Case = MatrixCase<sycl::half, sycl::half, float, 16, 16,
   16, 64, matrix_image_amd_gfx90a,
   matrix::matrix_type::fp16, matrix::matrix_type::fp32>;
using Amd90aBf16Case = MatrixCase<sycl::ext::oneapi::bfloat16,
   sycl::ext::oneapi::bfloat16, float, 16, 16, 16, 64,
   matrix_image_amd_gfx90a,
   matrix::matrix_type::bf16, matrix::matrix_type::fp32>;
using Amd90aInt8Case = MatrixCase<std::int8_t, std::int8_t, std::int32_t,
   16, 16, 16, 64, matrix_image_amd_gfx90a,
   matrix::matrix_type::sint8, matrix::matrix_type::sint32>;
using Amd90aFp64Case = MatrixCase<double, double, double, 16, 16, 4, 64,
   matrix_image_amd_gfx90a,
   matrix::matrix_type::fp64, matrix::matrix_type::fp64>;

using SpirvFp16WideCase = MatrixCase<sycl::half, sycl::half, float, 8, 16,
   16, 16, matrix_image_spirv,
   matrix::matrix_type::fp16, matrix::matrix_type::fp32>;
using SpirvBf16WideCase = MatrixCase<sycl::ext::oneapi::bfloat16,
   sycl::ext::oneapi::bfloat16, float, 8, 16, 16, 16,
   matrix_image_spirv,
   matrix::matrix_type::bf16, matrix::matrix_type::fp32>;
using SpirvTf32WideCase = MatrixCase<float, matrix::precision::tf32, float,
   8, 16, 8, 16, matrix_image_spirv,
   matrix::matrix_type::tf32, matrix::matrix_type::fp32>;
using SpirvInt8WideCase = MatrixCase<std::int8_t, std::int8_t, std::int32_t,
   8, 16, 32, 16, matrix_image_spirv,
   matrix::matrix_type::sint8, matrix::matrix_type::sint32>;

using SpirvFp16NarrowCase = MatrixCase<sycl::half, sycl::half, float, 8, 8,
   16, 16, matrix_image_spirv,
   matrix::matrix_type::fp16, matrix::matrix_type::fp32>;
using SpirvBf16NarrowCase = MatrixCase<sycl::ext::oneapi::bfloat16,
   sycl::ext::oneapi::bfloat16, float, 8, 8, 16, 16,
   matrix_image_spirv,
   matrix::matrix_type::bf16, matrix::matrix_type::fp32>;
using SpirvInt8NarrowCase = MatrixCase<std::int8_t, std::int8_t, std::int32_t,
   8, 8, 32, 16, matrix_image_spirv,
   matrix::matrix_type::sint8, matrix::matrix_type::sint32>;

using SpirvCpuFp16Case = MatrixCase<sycl::half, sycl::half, float, 16, 16,
   32, 16, matrix_image_spirv,
   matrix::matrix_type::fp16, matrix::matrix_type::fp32>;
using SpirvCpuBf16Case = MatrixCase<sycl::ext::oneapi::bfloat16,
   sycl::ext::oneapi::bfloat16, float, 16, 16, 32, 16,
   matrix_image_spirv,
   matrix::matrix_type::bf16, matrix::matrix_type::fp32>;
using SpirvCpuTf32Case = MatrixCase<float, matrix::precision::tf32, float,
   16, 16, 16, 16, matrix_image_spirv,
   matrix::matrix_type::tf32, matrix::matrix_type::fp32>;
using SpirvCpuInt8Case = MatrixCase<std::int8_t, std::int8_t, std::int32_t,
   16, 16, 64, 16, matrix_image_spirv,
   matrix::matrix_type::sint8, matrix::matrix_type::sint32>;

constexpr unsigned current_matrix_image =
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
#if SYCL_BENCH_COMPILED_CUDA_SM >= 80
   matrix_image_cuda_sm80;
#elif SYCL_BENCH_COMPILED_CUDA_SM >= 72
   matrix_image_cuda_sm72;
#else
   matrix_image_cuda_sm70;
#endif
#elif defined(__SYCL_DEVICE_ONLY__) && defined(__AMDGCN__) &&              \
   (defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__))
   matrix_image_amd_gfx11;
#elif defined(__SYCL_DEVICE_ONLY__) && defined(__AMDGCN__) &&              \
   defined(__gfx90a__)
   matrix_image_amd_gfx90a;
#elif defined(__SYCL_DEVICE_ONLY__) && defined(__AMDGCN__)
   0;
#else
   matrix_image_spirv;
#endif

template <typename Case>
inline constexpr bool matrix_case_supported_in_image =
   (Case::images & current_matrix_image) != 0;

bool matrix_dimension_supported(
   std::size_t requested, std::size_t exact, std::size_t maximum) {
   return exact != 0 ? requested == exact
                     : maximum != 0 && requested <= maximum;
}

constexpr int matrix_a_value(std::size_t row, std::size_t k) {
   return 1 + static_cast<int>((row + 2 * k) % 3);
}

constexpr int matrix_b_value(std::size_t k, std::size_t col) {
   return 1 + static_cast<int>((3 * k + col) % 4);
}

double matrix_reference(
   std::size_t row, std::size_t col, std::size_t size) {
   std::int64_t result = 0;
   for (std::size_t k = 0; k < size; ++k) {
      result += matrix_a_value(row, k) * matrix_b_value(k, col);
   }
   return static_cast<double>(result);
}

unsigned matrix_image_for_device(const sycl::device &device) {
   const auto backend = device.get_backend();
   if (backend == sycl::backend::ext_oneapi_cuda) {
      const std::string version =
         device.get_info<sycl::info::device::version>();
      try {
         const std::size_t dot = version.find('.');
         const int major = std::stoi(version.substr(0, dot));
         const int minor = dot == std::string::npos
            ? 0 : std::stoi(version.substr(dot + 1));
         int compute_capability = major * 10 + minor;
#ifdef SYCL_BENCH_COMPILED_CUDA_SM
         compute_capability = std::min(
            compute_capability, SYCL_BENCH_COMPILED_CUDA_SM);
#else
         return 0;
#endif
         if (compute_capability >= 80) {
            return matrix_image_cuda_sm80;
         }
         if (compute_capability >= 72) {
            return matrix_image_cuda_sm72;
         }
         if (compute_capability >= 70) {
            return matrix_image_cuda_sm70;
         }
      } catch (const std::exception &) {
      }
      return 0;
   }
   if (backend == sycl::backend::ext_oneapi_hip) {
      const std::string version =
         device.get_info<sycl::info::device::version>();
      if (version.rfind("gfx90a", 0) == 0) {
#ifdef SYCL_BENCH_COMPILED_AMD_GFX90A
         return matrix_image_amd_gfx90a;
#else
         return 0;
#endif
      }
      if (version.rfind("gfx1100", 0) == 0 ||
          version.rfind("gfx1101", 0) == 0 ||
          version.rfind("gfx1102", 0) == 0) {
#ifdef SYCL_BENCH_COMPILED_AMD_GFX11
         return matrix_image_amd_gfx11;
#else
         return 0;
#endif
      }
      return 0;
   }
   if (backend == sycl::backend::opencl ||
       backend == sycl::backend::ext_oneapi_level_zero) {
#ifdef SYCL_BENCH_COMPILED_SPIRV
      return matrix_image_spirv;
#else
      return 0;
#endif
   }
   return 0;
}

template <typename Case>
bool supports_matrix_gemm(const sycl::device &device) {
   if ((Case::images & matrix_image_for_device(device)) == 0) {
      return false;
   }
   const auto subgroup_sizes =
      device.get_info<sycl::info::device::sub_group_sizes>();
   if (std::find(subgroup_sizes.begin(), subgroup_sizes.end(),
          Case::subgroup) ==
       subgroup_sizes.end()) {
      return false;
   }
   try {
      const auto combinations = device.get_info<
         sycl::ext::oneapi::experimental::info::device::
            matrix_combinations>();
      return std::any_of(combinations.begin(), combinations.end(),
         [](const matrix::combination &c) {
            return matrix_dimension_supported(
                      Case::m, c.msize, c.max_msize) &&
                   matrix_dimension_supported(
                      Case::n, c.nsize, c.max_nsize) &&
                   matrix_dimension_supported(
                      Case::k, c.ksize, c.max_ksize) &&
                   c.atype == Case::input_type &&
                   c.btype == Case::input_type &&
                   c.ctype == Case::acc_matrix_type &&
                   c.dtype == Case::acc_matrix_type;
         });
   } catch (const sycl::exception &) {
      return false;
   }
}

template <typename Case> class MatrixGemmKernel;

template <typename Case>
sycl::event launch_matrix_gemm(sycl::queue &q,
   const typename Case::storage_type *a,
   const typename Case::storage_type *b, typename Case::acc_type *c,
   std::size_t size) {
   using Element = typename Case::element_type;
   using Acc = typename Case::acc_type;
   return q.parallel_for<MatrixGemmKernel<Case>>(
      sycl::nd_range<2>({size / (Case::m * Case::rows),
                         size / (Case::n * Case::cols) * Case::subgroup},
                        {1, Case::subgroup}),
      [=](sycl::nd_item<2> item)
#if !defined(__SYCL_DEVICE_ONLY__) ||                                    \
   (!defined(__NVPTX__) && !defined(__AMDGCN__))
         [[sycl::reqd_sub_group_size(Case::subgroup)]]
#endif
      {
#if defined(__SYCL_DEVICE_ONLY__)
         if constexpr (!matrix_case_supported_in_image<Case>) {
            // Backend-specific unsupported cases remain placeholders so one
            // fat binary can contain CUDA, SPIR-V and AMD images.
            (void)item;
            (void)a;
            (void)b;
            (void)c;
            (void)size;
         } else
#endif
         {
            const auto sg = item.get_sub_group();
            const std::size_t row =
               item.get_group(0) * Case::m * Case::rows;
            const std::size_t col =
               item.get_group(1) * Case::n * Case::cols;
            auto pa = sycl::address_space_cast<
               sycl::access::address_space::global_space,
               sycl::access::decorated::no>(a);
            auto pb = sycl::address_space_cast<
               sycl::access::address_space::global_space,
               sycl::access::decorated::no>(b);
            auto pc = sycl::address_space_cast<
               sycl::access::address_space::global_space,
               sycl::access::decorated::no>(c);

            matrix::joint_matrix<sycl::sub_group, Element, matrix::use::a,
               Case::m, Case::k, matrix::layout::row_major> ma[Case::rows];
            matrix::joint_matrix<sycl::sub_group, Element, matrix::use::b,
               Case::k, Case::n, matrix::layout::row_major> mb[Case::cols];
            matrix::joint_matrix<sycl::sub_group, Acc,
               matrix::use::accumulator, Case::m, Case::n>
               mc[Case::rows][Case::cols];
            for (std::size_t i = 0; i < Case::rows; ++i) {
               for (std::size_t j = 0; j < Case::cols; ++j) {
                  matrix::joint_matrix_fill(sg, mc[i][j], Acc{});
               }
            }
            for (std::size_t k = 0; k < size; k += Case::k) {
               for (std::size_t i = 0; i < Case::rows; ++i) {
                  matrix::joint_matrix_load(
                     sg, ma[i], pa + (row + i * Case::m) * size + k, size);
               }
               for (std::size_t j = 0; j < Case::cols; ++j) {
                  matrix::joint_matrix_load(
                     sg, mb[j], pb + k * size + col + j * Case::n, size);
               }
               for (std::size_t i = 0; i < Case::rows; ++i) {
                  for (std::size_t j = 0; j < Case::cols; ++j) {
                     matrix::joint_matrix_mad(
                        sg, mc[i][j], ma[i], mb[j], mc[i][j]);
                  }
               }
            }
            for (std::size_t i = 0; i < Case::rows; ++i) {
               for (std::size_t j = 0; j < Case::cols; ++j) {
                  matrix::joint_matrix_store(sg, mc[i][j],
                     pc + (row + i * Case::m) * size + col + j * Case::n,
                     size, matrix::layout::row_major);
               }
            }
         }
      });
}

template <typename Case>
MatrixRate matrix_gemm(sycl::queue &q, const sycl::device &device) {
   using Storage = typename Case::storage_type;
   using Acc = typename Case::acc_type;
   std::size_t size = 2048;
   const std::size_t budget =
      device.get_info<sycl::info::device::global_mem_size>() / 16;
   while (size > 256 &&
          (2 * sizeof(Storage) + sizeof(Acc)) * size * size > budget) {
      size /= 2;
   }

   const bool profiling = device.has(sycl::aspect::queue_profiling);
   sycl::queue mq = timing_queue(q.get_context(), device, profiling);
   UsmBuffer<Storage> a(mq, size * size), b(mq, size * size);
   UsmBuffer<Acc> c(mq, size * size);
   auto *pa = a.get();
   auto *pb = b.get();
   mq.parallel_for(sycl::range<1>(size * size), [=](sycl::id<1> id) {
      const std::size_t row = id[0] / size;
      const std::size_t col = id[0] % size;
      pa[id] = static_cast<Storage>(matrix_a_value(row, col));
      pb[id] = static_cast<Storage>(matrix_b_value(row, col));
   }).wait_and_throw();

   auto launch = [&] {
      return launch_matrix_gemm<Case>(mq, a.get(), b.get(), c.get(), size);
   };
   const double best_ns = best_kernel_ns(launch, profiling, 5);
   const std::array<std::array<std::size_t, 2>, 6> check_points{{
      {0, 0},
      {0, size - 1},
      {size / 3, size / 5},
      {size / 2, size / 2 + 1},
      {size - 1, 0},
      {size - 1, size - 1},
   }};
   for (const auto &point : check_points) {
      const std::size_t row = point[0];
      const std::size_t col = point[1];
      Acc check{};
      mq.memcpy(&check, c.get() + row * size + col, sizeof(check))
         .wait_and_throw();
      const double value = static_cast<double>(check);
      const double expected = matrix_reference(row, col, size);
      if (!std::isfinite(value) || std::abs(value - expected) > 0.5) {
         throw std::runtime_error("matrix GEMM verification failed at (" +
            std::to_string(row) + ", " + std::to_string(col) + ")");
      }
   }
   const double operations = 2.0 * size * size * size;
   return {size, operations / best_ns / 1.0e3};
}

template <typename Case> class MatrixIssueKernel;

template <typename Case>
double matrix_issue_rate(sycl::queue &q, const sycl::device &device) {
   using Storage = typename Case::storage_type;
   using Element = typename Case::element_type;
   using Acc = typename Case::acc_type;
   constexpr std::size_t accumulators = 4;
   const std::size_t groups = std::max<std::size_t>(256,
      16 * device.get_info<sycl::info::device::max_compute_units>());
   const bool profiling = device.has(sycl::aspect::queue_profiling);
   sycl::queue iq = timing_queue(q.get_context(), device, profiling);
   UsmBuffer<Storage> a(iq, Case::m * Case::k);
   UsmBuffer<Storage> b(iq, Case::k * Case::n);
   UsmBuffer<Acc> output(iq, groups * accumulators * Case::m * Case::n);
   const Storage input = [] {
      if constexpr (std::is_integral_v<Storage>) {
         return Storage{1};
      } else {
         return Storage{0.03125};
      }
   }();
   iq.fill(a.get(), input, Case::m * Case::k);
   iq.fill(b.get(), input, Case::k * Case::n).wait_and_throw();

   auto launch = [&](std::size_t iterations) {
      const auto *pa_raw = a.get();
      const auto *pb_raw = b.get();
      auto *out_raw = output.get();
      return iq.parallel_for<MatrixIssueKernel<Case>>(
         sycl::nd_range<1>(
            groups * Case::subgroup, Case::subgroup),
         [=](sycl::nd_item<1> item)
#if !defined(__SYCL_DEVICE_ONLY__) ||                                    \
   (!defined(__NVPTX__) && !defined(__AMDGCN__))
            [[sycl::reqd_sub_group_size(Case::subgroup)]]
#endif
         {
#if defined(__SYCL_DEVICE_ONLY__)
            if constexpr (!matrix_case_supported_in_image<Case>) {
               // See launch_matrix_gemm: unsupported backend matrix images
               // retain a placeholder kernel.
               (void)item;
               (void)iterations;
               (void)pa_raw;
               (void)pb_raw;
               (void)out_raw;
            } else
#endif
            {
               const auto sg = item.get_sub_group();
               auto pa = sycl::address_space_cast<
                  sycl::access::address_space::global_space,
                  sycl::access::decorated::no>(pa_raw);
               auto pb = sycl::address_space_cast<
                  sycl::access::address_space::global_space,
                  sycl::access::decorated::no>(pb_raw);
               auto pc = sycl::address_space_cast<
                  sycl::access::address_space::global_space,
                  sycl::access::decorated::no>(out_raw);
               matrix::joint_matrix<sycl::sub_group, Element, matrix::use::a,
                  Case::m, Case::k, matrix::layout::row_major> ma;
               matrix::joint_matrix<sycl::sub_group, Element, matrix::use::b,
                  Case::k, Case::n, matrix::layout::row_major> mb;
               matrix::joint_matrix<sycl::sub_group, Acc,
                  matrix::use::accumulator, Case::m, Case::n> mc[accumulators];
               matrix::joint_matrix_load(sg, ma, pa, Case::k);
               matrix::joint_matrix_load(sg, mb, pb, Case::n);
               for (std::size_t i = 0; i < accumulators; ++i) {
                  matrix::joint_matrix_fill(sg, mc[i], static_cast<Acc>(i));
               }
               for (std::size_t r = 0; r < iterations; ++r) {
                  for (std::size_t i = 0; i < accumulators; ++i) {
                     matrix::joint_matrix_mad(sg, mc[i], ma, mb, mc[i]);
                  }
               }
               const std::size_t base = item.get_group_linear_id() *
                  accumulators * Case::m * Case::n;
               for (std::size_t i = 0; i < accumulators; ++i) {
                  matrix::joint_matrix_store(sg, mc[i],
                     pc + base + i * Case::m * Case::n, Case::n,
                     matrix::layout::row_major);
               }
            }
         });
   };

   std::size_t iterations = 128;
   for (int attempt = 0; attempt < 5; ++attempt) {
      const double sample =
         elapsed_ms([&] { launch(iterations).wait_and_throw(); });
      if (sample >= 30.0 || iterations == (1u << 20)) {
         break;
      }
      const auto scale = static_cast<std::size_t>(
         std::clamp(80.0 / std::max(sample, 0.05), 2.0, 32.0));
      iterations = std::min<std::size_t>(1u << 20, iterations * scale);
   }
   const double ns =
      best_kernel_ns([&] { return launch(iterations); }, profiling);
   const std::array<std::array<std::size_t, 2>, 2> checks{{
      {0, 0},
      {groups - 1, accumulators - 1},
   }};
   for (const auto &position : checks) {
      const std::size_t group = position[0];
      const std::size_t accumulator = position[1];
      const std::size_t offset =
         (group * accumulators + accumulator) * Case::m * Case::n;
      Acc check{};
      iq.memcpy(&check, output.get() + offset, sizeof(check)).wait_and_throw();
      const double expected = static_cast<double>(accumulator) +
         static_cast<double>(iterations) * Case::k *
            static_cast<double>(input) * static_cast<double>(input);
      const double value = static_cast<double>(check);
      if (!std::isfinite(value) || std::abs(value - expected) > 0.5) {
         throw std::runtime_error(
            "matrix issue-rate verification failed");
      }
   }
   const double operations = static_cast<double>(groups) * iterations *
      accumulators * 2 * Case::m * Case::n * Case::k;
   return operations / ns / 1.0e3;
}

template <typename Case>
std::vector<BenchmarkMeasurement> benchmark_matrix_case(sycl::queue &q,
   const sycl::device &device, const char *name) {
   std::vector<BenchmarkMeasurement> results;
   const char *unit = std::is_integral_v<typename Case::storage_type>
      ? "TIOP/s" : "TFLOP/s";
   if (!supports_matrix_gemm<Case>(device)) {
      BenchmarkMeasurement result{"matrix", name};
      result.status = "unavailable";
      result.message = "unsupported by device";
      results.push_back(std::move(result));
      return results;
   }
   try {
      const auto rate = matrix_gemm<Case>(q, device);
      BenchmarkMeasurement result{"matrix", name};
      result.value = rate.tops;
      result.unit = unit;
      result.detail = std::to_string(rate.size) + " x " +
         std::to_string(rate.size) + " x " + std::to_string(rate.size) +
         "; tile " + std::to_string(Case::m) + " x " +
         std::to_string(Case::n) + " x " + std::to_string(Case::k) +
         "; subgroup " + std::to_string(Case::subgroup);
      results.push_back(std::move(result));
   } catch (const std::exception &e) {
      BenchmarkMeasurement result{"matrix", name};
      result.message = concise_error(e);
      result.status = result.message.find("verification failed") !=
            std::string::npos
         ? "error" : "unavailable";
      results.push_back(std::move(result));
      return results;
   }

   try {
      BenchmarkMeasurement issue{"matrix", std::string(name) + ".issue"};
      issue.value = matrix_issue_rate<Case>(q, device);
      issue.unit = unit;
      results.push_back(std::move(issue));
   } catch (const std::exception &e) {
      BenchmarkMeasurement issue{"matrix", std::string(name) + ".issue"};
      issue.message = concise_error(e);
      issue.status = issue.message.find("verification failed") !=
            std::string::npos
         ? "error" : "unavailable";
      results.push_back(std::move(issue));
   }
   return results;
}

template <typename Case>
void append_matrix_case(std::vector<BenchmarkMeasurement> &measurements,
   sycl::queue &q, const sycl::device &device, const char *name) {
   auto results = benchmark_matrix_case<Case>(q, device, name);
   measurements.insert(measurements.end(),
      std::make_move_iterator(results.begin()),
      std::make_move_iterator(results.end()));
}

template <typename First, typename... Rest>
void append_matrix_variants(std::vector<BenchmarkMeasurement> &measurements,
   sycl::queue &q, const sycl::device &device, const char *name) {
   if (supports_matrix_gemm<First>(device)) {
      append_matrix_case<First>(measurements, q, device, name);
   } else if constexpr (sizeof...(Rest) != 0) {
      append_matrix_variants<Rest...>(measurements, q, device, name);
   } else {
      BenchmarkMeasurement result{"matrix", name};
      result.status = "unavailable";
      result.message = "unsupported by device or compiled image";
      measurements.push_back(std::move(result));
   }
}

template <typename... Cases>
bool supports_any_matrix_variant(const sycl::device &device) {
   return (supports_matrix_gemm<Cases>(device) || ...);
}

struct MemoryRates {
   double read, write;
};

MemoryRates memory_bandwidth(
   sycl::queue &q, const sycl::device &device, std::size_t bytes) {
   const std::size_t count = bytes / sizeof(std::uint32_t);
   const std::size_t wg = std::min<std::size_t>(256,
      device.get_info<sycl::info::device::max_work_group_size>());
   const std::size_t workers = std::min(count, wg * std::max<std::size_t>(256,
      8 * device.get_info<sycl::info::device::max_compute_units>()));
   UsmBuffer<std::uint32_t> source(q, count), target(q, count);
   auto *src = source.get();
   auto *dst = target.get();
   q.fill(src, 0xa5a5a5a5u, count).wait_and_throw();

   auto read = [&] {
      return q.parallel_for(sycl::range<1>(workers), [=](sycl::id<1> id) {
         std::uint32_t sum = 0;
         for (std::size_t i = id; i < count; i += workers) {
            sum ^= src[i];
         }
         dst[id] = sum;
      });
   };
   auto write = [&] {
      return q.parallel_for(sycl::range<1>(workers), [=](sycl::id<1> id) {
         for (std::size_t i = id; i < count; i += workers) {
            dst[i] = static_cast<std::uint32_t>(i);
         }
      });
   };
   auto rate = [&](auto &&launch) {
      const double ms = best_ms([&] { launch().wait_and_throw(); }, 5);
      return gbps(count * sizeof(std::uint32_t), ms);
   };

   const double read_rate = rate(read);
   const std::array<std::size_t, 3> read_checks{
      0, workers / 2, workers - 1};
   for (const std::size_t id : read_checks) {
      std::uint32_t value = 0;
      q.memcpy(&value, dst + id, sizeof(value)).wait_and_throw();
      const std::size_t loads = 1 + (count - 1 - id) / workers;
      const std::uint32_t expected =
         loads % 2 == 0 ? 0u : 0xa5a5a5a5u;
      if (value != expected) {
         throw std::runtime_error("device-memory read verification failed");
      }
   }

   const double write_rate = rate(write);
   const std::array<std::size_t, 3> write_checks{
      0, count / 2, count - 1};
   for (const std::size_t id : write_checks) {
      std::uint32_t value = 0;
      q.memcpy(&value, dst + id, sizeof(value)).wait_and_throw();
      if (value != static_cast<std::uint32_t>(id)) {
         throw std::runtime_error("device-memory write verification failed");
      }
   }
   return {read_rate, write_rate};
}

struct TransferRates {
   double host_to_device, device_to_host, bidirectional;
};

TransferRates transfer_bandwidth(sycl::queue &q, const sycl::device &device,
   std::size_t bytes) {
   UsmBuffer<std::uint8_t> host(q, 2 * bytes, true);
   UsmBuffer<std::uint8_t> gpu(q, 2 * bytes);
   auto *h = host.get();
   auto *d = gpu.get();

   std::memset(h, 0x5a, bytes);
   q.memset(d, 0, bytes).wait_and_throw();
   const double h2d =
      best_ms([&] { q.memcpy(d, h, bytes).wait_and_throw(); }, 5);
   for (const std::size_t offset : {std::size_t{0}, bytes - 1}) {
      std::uint8_t value = 0;
      q.memcpy(&value, d + offset, sizeof(value)).wait_and_throw();
      if (value != 0x5a) {
         throw std::runtime_error(
            "host-to-device transfer verification failed");
      }
   }

   q.memset(d, 0xa5, bytes).wait_and_throw();
   std::memset(h, 0, bytes);
   const double d2h =
      best_ms([&] { q.memcpy(h, d, bytes).wait_and_throw(); }, 5);
   if (h[0] != 0xa5 || h[bytes - 1] != 0xa5) {
      throw std::runtime_error(
         "device-to-host transfer verification failed");
   }

   std::memset(h, 0x3c, bytes);
   std::memset(h + bytes, 0, bytes);
   q.memset(d, 0, bytes).wait_and_throw();
   q.memset(d + bytes, 0xc3, bytes).wait_and_throw();
   sycl::queue send(q.get_context(), device);
   sycl::queue receive(q.get_context(), device);
   const double both = best_ms(
      [&] {
         auto a = send.memcpy(d, h, bytes);
         auto b = receive.memcpy(h + bytes, d + bytes, bytes);
         a.wait_and_throw();
         b.wait_and_throw();
      },
      5);
   for (const std::size_t offset : {std::size_t{0}, bytes - 1}) {
      std::uint8_t value = 0;
      q.memcpy(&value, d + offset, sizeof(value)).wait_and_throw();
      if (value != 0x3c) {
         throw std::runtime_error(
            "bidirectional transfer verification failed");
      }
   }
   if (h[bytes] != 0xc3 || h[2 * bytes - 1] != 0xc3) {
      throw std::runtime_error(
         "bidirectional transfer verification failed");
   }
   return {gbps(bytes, h2d), gbps(bytes, d2h), gbps(2 * bytes, both)};
}

enum class StressProfile { compute, vram, mixed };

enum class StressComputeWorkload {
   fp32,
   fp64,
   matrix_fp16,
   matrix_bf16,
   matrix_tf32,
   matrix_int8,
   matrix_fp64
};

constexpr std::size_t default_stress_chunk_bytes = 512ull << 20;

struct StressOptions {
   std::chrono::seconds duration{60};
   std::chrono::seconds report_interval{5};
   StressProfile profile = StressProfile::mixed;
   StressComputeWorkload compute_workload = StressComputeWorkload::fp32;
   unsigned memory_percent = 50;
   std::size_t chunk_size_bytes = default_stress_chunk_bytes;
   std::uint32_t seed = 0x6d2b79f5u;
   bool parallel = true;
   double minimum_compute_rate = -1.0;
   double minimum_vram_rate = -1.0;
   double maximum_slowdown = -1.0;
};

volatile std::sig_atomic_t stress_stop_requested = 0;

void request_stress_stop(int) { stress_stop_requested = 1; }

const char *stress_profile_name(StressProfile profile) {
   switch (profile) {
   case StressProfile::compute:
      return "compute";
   case StressProfile::vram:
      return "vram";
   case StressProfile::mixed:
      return "mixed";
   }
   return "unknown";
}

const char *stress_compute_workload_name(StressComputeWorkload workload) {
   switch (workload) {
   case StressComputeWorkload::fp32:
      return "fp32";
   case StressComputeWorkload::fp64:
      return "fp64";
   case StressComputeWorkload::matrix_fp16:
      return "matrix-fp16";
   case StressComputeWorkload::matrix_bf16:
      return "matrix-bf16";
   case StressComputeWorkload::matrix_tf32:
      return "matrix-tf32";
   case StressComputeWorkload::matrix_int8:
      return "matrix-int8";
   case StressComputeWorkload::matrix_fp64:
      return "matrix-fp64";
   }
   return "unknown";
}

const char *stress_compute_unit(StressComputeWorkload workload) {
   return workload == StressComputeWorkload::matrix_int8 ? "TIOP/s"
                                                         : "TFLOP/s";
}

class AsyncErrorState {
public:
   void capture(sycl::exception_list errors) {
      std::lock_guard<std::mutex> lock(mutex_);
      for (const auto &error : errors) {
         try {
            std::rethrow_exception(error);
         } catch (const std::exception &e) {
            errors_.push_back(concise_error(e));
         }
      }
   }

   void throw_if_any() {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!errors_.empty()) {
         const std::string message = errors_.front();
         errors_.clear();
         throw std::runtime_error("asynchronous SYCL error: " + message);
      }
   }

private:
   std::mutex mutex_;
   std::vector<std::string> errors_;
};

double completed_event_ms(sycl::event &event, bool profiling,
   Clock::time_point host_begin) {
   event.wait_and_throw();
   if (profiling) {
      try {
         const auto begin = event.get_profiling_info<
            sycl::info::event_profiling::command_start>();
         const auto end = event.get_profiling_info<
            sycl::info::event_profiling::command_end>();
         if (end > begin) {
            return static_cast<double>(end - begin) / 1.0e6;
         }
      } catch (const sycl::exception &) {
         // Fall back to host timing if this backend exposes profiling but
         // does not provide usable timestamps for the event.
      }
   }
   return std::chrono::duration<double, std::milli>(
      Clock::now() - host_begin).count();
}

class StressComputeRunner {
public:
   virtual ~StressComputeRunner() = default;
   virtual double run() = 0;
   virtual std::uint32_t verify() = 0;
};

template <typename T>
class StressScalarCompute final : public StressComputeRunner {
public:
   StressScalarCompute(
      sycl::queue &q, const sycl::device &device, bool profiling)
      : q_(q), profiling_(profiling),
        items_(std::min<std::size_t>(256,
                  device.get_info<sycl::info::device::max_work_group_size>()) *
           std::max<std::size_t>(256,
              8 * device.get_info<sycl::info::device::max_compute_units>())),
        input_(q, items_), output_(q, items_), errors_(q, 1) {
      q_.fill(input_.get(), T{1}, items_).wait_and_throw();

      for (int attempt = 0; attempt < 5; ++attempt) {
         const auto begin = Clock::now();
         auto event = launch(iterations_);
         const double ms = completed_event_ms(event, profiling_, begin);
         if (ms >= 40.0 || iterations_ == (1u << 20)) {
            break;
         }
         const auto scale = static_cast<std::size_t>(
            std::clamp(80.0 / std::max(ms, 0.05), 2.0, 32.0));
         iterations_ =
            std::min<std::size_t>(1u << 20, iterations_ * scale);
      }
      expected_ = reference_result(iterations_);
   }

   double run() override {
      const auto begin = Clock::now();
      auto event = launch(iterations_);
      const double ms = completed_event_ms(event, profiling_, begin);
      const double operations = static_cast<double>(items_) * iterations_ *
         accumulators * 2;
      return operations / (ms * 1.0e9);
   }

   std::uint32_t verify() override {
      q_.fill(errors_.get(), 0u, 1).wait_and_throw();
      const T expected = expected_;
      const T tolerance =
         std::max(T{1.0e-3}, sycl::fabs(expected) * T{1.0e-4});
      const T *out = output_.get();
      std::uint32_t *error_count = errors_.get();
      q_.parallel_for(sycl::range<1>(items_), [=](sycl::id<1> id) {
         const T value = out[id];
         if (!sycl::isfinite(value) ||
             sycl::fabs(value - expected) > tolerance) {
            sycl::atomic_ref<std::uint32_t,
               sycl::memory_order::relaxed, sycl::memory_scope::device,
               sycl::access::address_space::global_space>(*error_count)
               .fetch_add(1);
         }
      }).wait_and_throw();
      std::uint32_t result = 0;
      q_.memcpy(&result, errors_.get(), sizeof(result)).wait_and_throw();
      return result;
   }

private:
   static constexpr std::size_t accumulators = 16;

   sycl::event launch(std::size_t iterations) {
      const T *in = input_.get();
      T *out = output_.get();
      return q_.parallel_for(sycl::range<1>(items_), [=](sycl::id<1> id) {
         const T x = in[id];
         T acc[accumulators];
#pragma unroll
         for (std::size_t lane = 0; lane < accumulators; ++lane) {
            acc[lane] = x + static_cast<T>(lane + 1);
         }
         for (std::size_t i = 0; i < iterations; ++i) {
#pragma unroll
            for (std::size_t lane = 0; lane < accumulators; ++lane) {
               acc[lane] = sycl::fma(acc[lane], T{0.999}, T{0.001});
            }
         }
         T sum{};
#pragma unroll
         for (std::size_t lane = 0; lane < accumulators; ++lane) {
            sum += acc[lane];
         }
         out[id] = sum;
      });
   }

   static T reference_result(std::size_t iterations) {
      T acc[accumulators];
      for (std::size_t lane = 0; lane < accumulators; ++lane) {
         acc[lane] = T{1} + static_cast<T>(lane + 1);
      }
      for (std::size_t i = 0; i < iterations; ++i) {
         for (T &value : acc) {
            value = std::fma(value, T{0.999}, T{0.001});
         }
      }
      T sum{};
      for (const T value : acc) {
         sum += value;
      }
      return sum;
   }

   sycl::queue &q_;
   bool profiling_;
   std::size_t items_;
   std::size_t iterations_ = 256;
   T expected_{};
   UsmBuffer<T> input_;
   UsmBuffer<T> output_;
   UsmBuffer<std::uint32_t> errors_;
};

template <typename Case> class StressMatrixKernel;

template <typename Case>
class StressMatrixCompute final : public StressComputeRunner {
public:
   using Storage = typename Case::storage_type;
   using Element = typename Case::element_type;
   using Acc = typename Case::acc_type;

   StressMatrixCompute(
      sycl::queue &q, const sycl::device &device, bool profiling)
      : q_(q), profiling_(profiling),
        groups_(std::max<std::size_t>(256,
           16 * device.get_info<sycl::info::device::max_compute_units>())),
        input_(matrix_input()), a_(q, Case::m * Case::k),
        b_(q, Case::k * Case::n),
        output_(q, groups_ * accumulators * Case::m * Case::n),
        errors_(q, 1) {
      q_.fill(a_.get(), input_, Case::m * Case::k);
      q_.fill(b_.get(), input_, Case::k * Case::n);
      q_.fill(output_.get(), Acc{}, output_count()).wait_and_throw();

      for (int attempt = 0; attempt < 5; ++attempt) {
         const auto begin = Clock::now();
         auto event = launch(iterations_);
         const double ms = completed_event_ms(event, profiling_, begin);
         if (ms >= 40.0 || iterations_ == (1u << 20)) {
            break;
         }
         const auto scale = static_cast<std::size_t>(
            std::clamp(80.0 / std::max(ms, 0.05), 2.0, 32.0));
         iterations_ =
            std::min<std::size_t>(1u << 20, iterations_ * scale);
      }
   }

   double run() override {
      const auto begin = Clock::now();
      auto event = launch(iterations_);
      const double ms = completed_event_ms(event, profiling_, begin);
      const double operations = static_cast<double>(groups_) * iterations_ *
         accumulators * 2 * Case::m * Case::n * Case::k;
      return operations / (ms * 1.0e9);
   }

   std::uint32_t verify() override {
      q_.fill(errors_.get(), 0u, 1).wait_and_throw();
      const Acc *out = output_.get();
      std::uint32_t *error_count = errors_.get();
      const Storage input = input_;
      const std::size_t iterations = iterations_;
      constexpr std::size_t matrix_elements = Case::m * Case::n;
      q_.parallel_for(sycl::range<1>(output_count()), [=](sycl::id<1> id) {
         const std::size_t accumulator =
            (id[0] / matrix_elements) % accumulators;
         const Acc expected = static_cast<Acc>(accumulator) +
            static_cast<Acc>(iterations * Case::k) *
               static_cast<Acc>(input) * static_cast<Acc>(input);
         const Acc value = out[id];
         bool incorrect;
         if constexpr (std::is_integral_v<Acc>) {
            incorrect = value != expected;
         } else {
            const Acc tolerance =
               std::max(Acc{1.0e-4}, sycl::fabs(expected) * Acc{1.0e-5});
            incorrect = !sycl::isfinite(value) ||
               sycl::fabs(value - expected) > tolerance;
         }
         if (incorrect) {
            sycl::atomic_ref<std::uint32_t,
               sycl::memory_order::relaxed, sycl::memory_scope::device,
               sycl::access::address_space::global_space>(*error_count)
               .fetch_add(1);
         }
      }).wait_and_throw();
      std::uint32_t result = 0;
      q_.memcpy(&result, errors_.get(), sizeof(result)).wait_and_throw();
      return result;
   }

private:
   static constexpr std::size_t accumulators = 4;

   static Storage matrix_input() {
      if constexpr (std::is_integral_v<Storage>) {
         return Storage{1};
      } else {
         return Storage{0.03125};
      }
   }

   std::size_t output_count() const {
      return groups_ * accumulators * Case::m * Case::n;
   }

   sycl::event launch(std::size_t iterations) {
      const Storage *pa_raw = a_.get();
      const Storage *pb_raw = b_.get();
      Acc *out_raw = output_.get();
      return q_.parallel_for<StressMatrixKernel<Case>>(
         sycl::nd_range<1>(
            groups_ * Case::subgroup, Case::subgroup),
         [=](sycl::nd_item<1> item)
#if !defined(__SYCL_DEVICE_ONLY__) ||                                    \
   (!defined(__NVPTX__) && !defined(__AMDGCN__))
            [[sycl::reqd_sub_group_size(Case::subgroup)]]
#endif
         {
#if defined(__SYCL_DEVICE_ONLY__)
            if constexpr (!matrix_case_supported_in_image<Case>) {
               (void)item;
               (void)iterations;
               (void)pa_raw;
               (void)pb_raw;
               (void)out_raw;
            } else
#endif
            {
               const auto sg = item.get_sub_group();
               auto pa = sycl::address_space_cast<
                  sycl::access::address_space::global_space,
                  sycl::access::decorated::no>(pa_raw);
               auto pb = sycl::address_space_cast<
                  sycl::access::address_space::global_space,
                  sycl::access::decorated::no>(pb_raw);
               auto pc = sycl::address_space_cast<
                  sycl::access::address_space::global_space,
                  sycl::access::decorated::no>(out_raw);
               matrix::joint_matrix<sycl::sub_group, Element, matrix::use::a,
                  Case::m, Case::k, matrix::layout::row_major> ma;
               matrix::joint_matrix<sycl::sub_group, Element, matrix::use::b,
                  Case::k, Case::n, matrix::layout::row_major> mb;
               matrix::joint_matrix<sycl::sub_group, Acc,
                  matrix::use::accumulator, Case::m, Case::n> mc[accumulators];
               matrix::joint_matrix_load(sg, ma, pa, Case::k);
               matrix::joint_matrix_load(sg, mb, pb, Case::n);
               for (std::size_t i = 0; i < accumulators; ++i) {
                  matrix::joint_matrix_fill(sg, mc[i], static_cast<Acc>(i));
               }
               for (std::size_t r = 0; r < iterations; ++r) {
                  for (std::size_t i = 0; i < accumulators; ++i) {
                     matrix::joint_matrix_mad(sg, mc[i], ma, mb, mc[i]);
                  }
               }
               const std::size_t base = item.get_group_linear_id() *
                  accumulators * Case::m * Case::n;
               for (std::size_t i = 0; i < accumulators; ++i) {
                  matrix::joint_matrix_store(sg, mc[i],
                     pc + base + i * Case::m * Case::n, Case::n,
                     matrix::layout::row_major);
               }
            }
         });
   }

   sycl::queue &q_;
   bool profiling_;
   std::size_t groups_;
   std::size_t iterations_ = 128;
   Storage input_;
   UsmBuffer<Storage> a_;
   UsmBuffer<Storage> b_;
   UsmBuffer<Acc> output_;
   UsmBuffer<std::uint32_t> errors_;
};

template <typename Case>
std::unique_ptr<StressComputeRunner> make_stress_matrix_compute(
   sycl::queue &q, const sycl::device &device, bool profiling,
   StressComputeWorkload workload) {
   if (!supports_matrix_gemm<Case>(device)) {
      throw std::runtime_error(std::string("compute workload ") +
         stress_compute_workload_name(workload) +
         " is not supported by device");
   }
   auto result =
      std::make_unique<StressMatrixCompute<Case>>(q, device, profiling);
   const std::uint32_t errors = result->verify();
   if (errors != 0) {
      throw std::runtime_error(std::string("compute workload ") +
         stress_compute_workload_name(workload) +
         " failed initial verification; device image may not support it");
   }
   return result;
}

template <typename First, typename... Rest>
std::unique_ptr<StressComputeRunner> make_stress_matrix_variants(
   sycl::queue &q, const sycl::device &device, bool profiling,
   StressComputeWorkload workload) {
   if (supports_matrix_gemm<First>(device)) {
      return make_stress_matrix_compute<First>(
         q, device, profiling, workload);
   }
   if constexpr (sizeof...(Rest) != 0) {
      return make_stress_matrix_variants<Rest...>(
         q, device, profiling, workload);
   }
   throw std::runtime_error(std::string("compute workload ") +
      stress_compute_workload_name(workload) +
      " is not supported by device or compiled image");
}

std::unique_ptr<StressComputeRunner> make_stress_compute(
   sycl::queue &q, const sycl::device &device, bool profiling,
   StressComputeWorkload workload) {
   switch (workload) {
   case StressComputeWorkload::fp32:
      return std::make_unique<StressScalarCompute<float>>(
         q, device, profiling);
   case StressComputeWorkload::fp64:
      if (!device.has(sycl::aspect::fp64)) {
         throw std::runtime_error(
            "compute workload fp64 is not supported by device");
      }
      return std::make_unique<StressScalarCompute<double>>(
         q, device, profiling);
   case StressComputeWorkload::matrix_fp16:
      return make_stress_matrix_variants<SpirvFp16WideCase,
         SpirvFp16NarrowCase, SpirvCpuFp16Case, Fp16Case,
         Amd90aFp16Case>(q, device, profiling, workload);
   case StressComputeWorkload::matrix_bf16:
      return make_stress_matrix_variants<SpirvBf16WideCase,
         SpirvBf16NarrowCase, SpirvCpuBf16Case, Bf16Case,
         Amd90aBf16Case>(q, device, profiling, workload);
   case StressComputeWorkload::matrix_tf32:
      return make_stress_matrix_variants<SpirvTf32WideCase,
         SpirvCpuTf32Case, Tf32Case>(q, device, profiling, workload);
   case StressComputeWorkload::matrix_int8:
      return make_stress_matrix_variants<SpirvInt8WideCase,
         SpirvInt8NarrowCase, SpirvCpuInt8Case, Int8Case,
         Amd90aInt8Case>(q, device, profiling, workload);
   case StressComputeWorkload::matrix_fp64:
      return make_stress_matrix_variants<Fp64Case, Amd90aFp64Case>(
         q, device, profiling, workload);
   }
   throw std::runtime_error("unknown stress compute workload");
}

std::uint32_t stress_pattern(std::size_t index, std::uint32_t seed) {
   const std::uint64_t wide = static_cast<std::uint64_t>(index);
   std::uint32_t value = static_cast<std::uint32_t>(wide ^ (wide >> 32)) ^ seed;
   value ^= value >> 16;
   value *= 0x7feb352du;
   value ^= value >> 15;
   value *= 0x846ca68bu;
   value ^= value >> 16;
   return value;
}

struct VramStressResult {
   double gbps;
   std::uint64_t errors;
   std::size_t first_error;
};

template <typename T> class ChunkedUsmBuffer {
public:
   struct Chunk {
      T *pointer;
      std::size_t count;
      std::size_t base;
   };

   ChunkedUsmBuffer(sycl::queue &q, std::size_t total_count,
      std::size_t maximum_chunk_bytes)
      : q_(q), maximum_chunk_bytes_(maximum_chunk_bytes) {
      const std::size_t maximum_chunk_count =
         maximum_chunk_bytes_ / sizeof(T);
      if (maximum_chunk_count == 0) {
         throw std::invalid_argument("chunk size is smaller than one element");
      }
      std::size_t allocated = 0;
      try {
         while (allocated < total_count) {
            const std::size_t count =
               std::min(maximum_chunk_count, total_count - allocated);
            T *pointer = sycl::malloc_device<T>(count, q_);
            if (!pointer) {
               throw std::runtime_error(
                  "device allocation failed after " +
                  std::to_string(allocated * sizeof(T) / (1024 * 1024)) +
                  " MiB of " +
                  std::to_string(total_count * sizeof(T) / (1024 * 1024)) +
                  " MiB");
            }
            chunks_.push_back({pointer, count, allocated});
            allocated += count;
         }
      } catch (...) {
         release();
         throw;
      }
   }

   ~ChunkedUsmBuffer() { release(); }
   ChunkedUsmBuffer(const ChunkedUsmBuffer &) = delete;
   ChunkedUsmBuffer &operator=(const ChunkedUsmBuffer &) = delete;

   const std::vector<Chunk> &chunks() const { return chunks_; }
   std::size_t maximum_chunk_bytes() const { return maximum_chunk_bytes_; }

private:
   void release() {
      for (const auto &chunk : chunks_) {
         sycl::free(chunk.pointer, q_);
      }
      chunks_.clear();
   }

   sycl::queue &q_;
   std::size_t maximum_chunk_bytes_;
   std::vector<Chunk> chunks_;
};

class StressVram {
public:
   StressVram(sycl::queue &q, const sycl::device &device, unsigned percent,
      std::size_t chunk_size_bytes)
      : q_(q), bytes_(allocation_size(device, percent)),
        count_(bytes_ / sizeof(std::uint32_t)),
        workers_(std::min(count_,
           std::min<std::size_t>(256,
              device.get_info<sycl::info::device::max_work_group_size>()) *
              std::max<std::size_t>(256,
                 8 * device.get_info<sycl::info::device::max_compute_units>()))),
        data_(q, count_, effective_chunk_size(device, chunk_size_bytes)),
        worker_errors_(q, workers_),
        worker_first_(q, workers_), host_errors_(workers_),
        host_first_(workers_) {}

   std::size_t bytes() const { return bytes_; }
   std::size_t chunk_count() const { return data_.chunks().size(); }
   std::size_t chunk_bytes() const {
      return data_.maximum_chunk_bytes();
   }

   VramStressResult run(std::uint32_t seed) {
      const auto begin = Clock::now();
      for (const auto &chunk : data_.chunks()) {
         std::uint32_t *data = chunk.pointer;
         const std::size_t count = chunk.count;
         const std::size_t base = chunk.base;
         const std::size_t workers = std::min(workers_, count);
         q_.parallel_for(sycl::range<1>(workers), [=](sycl::id<1> id) {
            for (std::size_t i = id; i < count; i += workers) {
               data[i] = stress_pattern(base + i, seed);
            }
         }).wait_and_throw();
      }

      std::uint32_t *errors = worker_errors_.get();
      std::size_t *first = worker_first_.get();
      std::uint64_t total_errors = 0;
      std::size_t first_error = count_;
      for (const auto &chunk : data_.chunks()) {
         std::uint32_t *data = chunk.pointer;
         const std::size_t count = chunk.count;
         const std::size_t base = chunk.base;
         const std::size_t workers = std::min(workers_, count);
         q_.parallel_for(sycl::range<1>(workers), [=](sycl::id<1> id) {
            std::uint32_t local_errors = 0;
            std::size_t local_first = count;
            for (std::size_t i = id; i < count; i += workers) {
               if (data[i] != stress_pattern(base + i, seed)) {
                  ++local_errors;
                  if (local_first == count) {
                     local_first = i;
                  }
               }
            }
            errors[id] = local_errors;
            first[id] = local_first;
         }).wait_and_throw();
         q_.memcpy(host_errors_.data(), worker_errors_.get(),
            workers * sizeof(std::uint32_t)).wait_and_throw();
         q_.memcpy(host_first_.data(), worker_first_.get(),
            workers * sizeof(std::size_t)).wait_and_throw();
         for (std::size_t i = 0; i < workers; ++i) {
            total_errors += host_errors_[i];
            if (host_first_[i] != count) {
               first_error =
                  std::min(first_error, base + host_first_[i]);
            }
         }
      }
      const double ms = std::chrono::duration<double, std::milli>(
         Clock::now() - begin).count();
      return {gbps(2 * bytes_, ms), total_errors, first_error};
   }

private:
   static std::size_t effective_chunk_size(
      const sycl::device &device, std::size_t requested) {
      const std::size_t device_limit =
         device.get_info<sycl::info::device::max_mem_alloc_size>();
      const std::size_t limited =
         device_limit == 0 ? requested : std::min(requested, device_limit);
      return std::max(sizeof(std::uint32_t),
         limited / sizeof(std::uint32_t) * sizeof(std::uint32_t));
   }

   static std::size_t allocation_size(
      const sycl::device &device, unsigned percent) {
      constexpr std::size_t alignment = 4096;
      constexpr std::size_t minimum = 4ull << 20;
      constexpr std::size_t shared_memory_cap = 8ull << 30;
      std::size_t capacity =
         device.get_info<sycl::info::device::global_mem_size>();
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
      const bool unified_memory =
         device.get_info<sycl::info::device::host_unified_memory>();
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
      if (unified_memory) {
         // Integrated devices often report most of system RAM as global
         // memory. Keep the default stress allocation from consuming half of
         // the machine's physical memory.
         capacity = std::min(capacity, shared_memory_cap);
      }
      const std::size_t requested = static_cast<std::size_t>(
         static_cast<long double>(capacity) * percent / 100.0L);
      return std::max(minimum, requested / alignment * alignment);
   }

   sycl::queue &q_;
   std::size_t bytes_;
   std::size_t count_;
   std::size_t workers_;
   ChunkedUsmBuffer<std::uint32_t> data_;
   UsmBuffer<std::uint32_t> worker_errors_;
   UsmBuffer<std::size_t> worker_first_;
   std::vector<std::uint32_t> host_errors_;
   std::vector<std::size_t> host_first_;
};

struct StressInterval {
   std::size_t compute_batches = 0;
   double compute_sum = 0.0;
   double compute_min = std::numeric_limits<double>::max();
   std::size_t vram_passes = 0;
   double vram_sum = 0.0;
   double vram_min = std::numeric_limits<double>::max();
};

struct RateSummary {
   std::size_t samples = 0;
   double average = 0.0;
   double minimum = 0.0;
   double p95 = 0.0;
};

class RateAccumulator {
public:
   void add(double value) { values_.push_back(value); }
   bool empty() const { return values_.empty(); }

   RateSummary summary() const {
      if (values_.empty()) {
         return {};
      }
      std::vector<double> sorted = values_;
      std::sort(sorted.begin(), sorted.end());
      double sum = 0.0;
      for (const double value : values_) {
         sum += value;
      }
      const std::size_t p95_index = static_cast<std::size_t>(
         std::ceil(0.95 * sorted.size())) - 1;
      return {values_.size(), sum / values_.size(), sorted.front(),
         sorted[p95_index]};
   }

private:
   std::vector<double> values_;
};

struct StressSummary {
   DeviceEntry entry;
   std::string timestamp;
   std::string profile;
   std::string compute_workload;
   std::string compute_unit;
   std::uint64_t requested_seconds = 0;
   unsigned memory_percent = 0;
   std::uint32_t seed = 0;
   std::string status = "pass";
   std::string message;
   double elapsed_seconds = 0.0;
   std::size_t memory_bytes = 0;
   std::size_t memory_chunks = 0;
   std::size_t memory_max_chunk_bytes = 0;
   double compute_slowdown = 0.0;
   double vram_slowdown = 0.0;
   RateSummary compute;
   RateSummary vram;
};

StressSummary make_stress_summary(const DeviceEntry &entry,
   const StressOptions &options) {
   StressSummary summary{entry};
   summary.timestamp = utc_timestamp();
   summary.profile = stress_profile_name(options.profile);
   summary.compute_workload =
      stress_compute_workload_name(options.compute_workload);
   summary.compute_unit = stress_compute_unit(options.compute_workload);
   summary.requested_seconds = options.duration.count();
   summary.memory_percent = options.memory_percent;
   summary.seed = options.seed;
   return summary;
}

using StressProgress = std::function<void(const DeviceEntry &,
   Clock::duration, const StressInterval &)>;

std::string stress_elapsed(Clock::duration elapsed) {
   const auto seconds =
      std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
   const auto hours = seconds / 3600;
   const auto minutes = seconds / 60 % 60;
   const auto remainder = seconds % 60;
   std::ostringstream out;
   out << std::setfill('0') << std::setw(2) << hours << ':'
       << std::setw(2) << minutes << ':' << std::setw(2) << remainder;
   return out.str();
}

StressSummary stress_device(const DeviceEntry &entry,
   const StressOptions &options, const StressProgress &progress,
   const std::atomic<bool> &abort_requested) {
   const sycl::device &device = entry.device;
   if (!device.has(sycl::aspect::usm_device_allocations)) {
      throw std::runtime_error("device USM is required");
   }
   AsyncErrorState async_errors;
   const bool profiling = device.has(sycl::aspect::queue_profiling);
   const sycl::property_list properties{
      sycl::property::queue::in_order{},
      sycl::property::queue::enable_profiling{}};
   sycl::queue q(device,
      [&](sycl::exception_list errors) { async_errors.capture(errors); },
      profiling ? properties
                : sycl::property_list{sycl::property::queue::in_order{}});

   std::unique_ptr<StressComputeRunner> compute;
   std::unique_ptr<StressVram> vram;
   if (options.profile != StressProfile::vram) {
      compute = make_stress_compute(
         q, device, profiling, options.compute_workload);
   }
   if (options.profile != StressProfile::compute) {
      vram = std::make_unique<StressVram>(
         q, device, options.memory_percent, options.chunk_size_bytes);
   }

   const auto started = Clock::now();
   const auto deadline = started + options.duration;
   auto next_report = started + options.report_interval;
   StressInterval interval;
   RateAccumulator compute_rates;
   RateAccumulator vram_rates;
   double compute_baseline = 0.0;
   double compute_worst_interval = std::numeric_limits<double>::max();
   double vram_baseline = 0.0;
   double vram_worst_interval = std::numeric_limits<double>::max();
   std::uint32_t seed = options.seed;

   const auto report_interval = [&](Clock::duration elapsed) {
      if (interval.compute_batches != 0) {
         const double average =
            interval.compute_sum / interval.compute_batches;
         if (compute_baseline == 0.0) {
            compute_baseline = average;
         }
         compute_worst_interval =
            std::min(compute_worst_interval, average);
      }
      if (interval.vram_passes != 0) {
         const double average = interval.vram_sum / interval.vram_passes;
         if (vram_baseline == 0.0) {
            vram_baseline = average;
         }
         vram_worst_interval = std::min(vram_worst_interval, average);
      }
      progress(entry, elapsed, interval);
      interval = StressInterval{};
   };

   const auto run_compute_batch = [&] {
         const double rate = compute->run();
         const std::uint32_t errors = compute->verify();
         async_errors.throw_if_any();
         if (errors != 0) {
            throw std::runtime_error("compute verification found " +
               std::to_string(errors) + " incorrect outputs");
         }
         ++interval.compute_batches;
         interval.compute_sum += rate;
         interval.compute_min = std::min(interval.compute_min, rate);
         compute_rates.add(rate);
   };
   const auto run_vram_pass = [&] {
         seed = seed * 1664525u + 1013904223u;
         const auto result = vram->run(seed);
         async_errors.throw_if_any();
         if (result.errors != 0) {
            throw std::runtime_error("VRAM verification found " +
               std::to_string(result.errors) +
               " errors; first byte offset " +
               std::to_string(result.first_error * sizeof(std::uint32_t)));
         }
         ++interval.vram_passes;
         interval.vram_sum += result.gbps;
         interval.vram_min = std::min(interval.vram_min, result.gbps);
         vram_rates.add(result.gbps);
   };

   bool next_compute = true;
   while (!stress_stop_requested && !abort_requested.load()) {
      const bool compute_missing = compute && compute_rates.empty();
      const bool vram_missing = vram && vram_rates.empty();
      if (Clock::now() >= deadline && !compute_missing && !vram_missing) {
         break;
      }

      if (compute && vram) {
         if (compute_missing) {
            run_compute_batch();
            next_compute = false;
         } else if (vram_missing) {
            run_vram_pass();
            next_compute = true;
         } else if (next_compute) {
            run_compute_batch();
            next_compute = false;
         } else {
            run_vram_pass();
            next_compute = true;
         }
      } else if (compute) {
         run_compute_batch();
      } else {
         run_vram_pass();
      }

      const auto now = Clock::now();
      if (now >= next_report) {
         report_interval(now - started);
         do {
            next_report += options.report_interval;
         } while (next_report <= now);
      }
   }

   if (interval.compute_batches != 0 || interval.vram_passes != 0) {
      report_interval(Clock::now() - started);
   }
   StressSummary summary = make_stress_summary(entry, options);
   summary.status = stress_stop_requested ? "interrupted"
      : abort_requested.load() ? "aborted" : "pass";
   summary.elapsed_seconds = std::chrono::duration<double>(
      Clock::now() - started).count();
   summary.memory_bytes = vram ? vram->bytes() : 0;
   summary.memory_chunks = vram ? vram->chunk_count() : 0;
   summary.memory_max_chunk_bytes = vram ? vram->chunk_bytes() : 0;
   summary.compute = compute_rates.summary();
   summary.vram = vram_rates.summary();
   if (compute_baseline > 0.0 &&
       compute_worst_interval != std::numeric_limits<double>::max()) {
      summary.compute_slowdown = std::max(0.0,
         (compute_baseline - compute_worst_interval) /
            compute_baseline * 100.0);
   }
   if (vram_baseline > 0.0 &&
       vram_worst_interval != std::numeric_limits<double>::max()) {
      summary.vram_slowdown = std::max(0.0,
         (vram_baseline - vram_worst_interval) / vram_baseline * 100.0);
   }
   const auto fail = [&](const std::string &message) {
      summary.status = "fail";
      if (!summary.message.empty()) {
         summary.message += "; ";
      }
      summary.message += message;
   };
   if (summary.status == "pass" && compute && summary.compute.samples == 0) {
      fail("compute workload produced no samples");
   }
   if (summary.status == "pass" && vram && summary.vram.samples == 0) {
      fail("VRAM workload produced no samples");
   }
   if (summary.status == "pass" && options.minimum_compute_rate >= 0.0 &&
       summary.compute.samples != 0 &&
       summary.compute.average < options.minimum_compute_rate) {
      fail("compute average is below the required rate");
   }
   if (summary.status == "pass" && options.minimum_vram_rate >= 0.0 &&
       summary.vram.samples != 0 &&
       summary.vram.average < options.minimum_vram_rate) {
      fail("VRAM average is below the required rate");
   }
   if (summary.status == "pass" && options.maximum_slowdown >= 0.0 &&
       ((summary.compute.samples != 0 &&
            summary.compute_slowdown > options.maximum_slowdown) ||
          (summary.vram.samples != 0 &&
            summary.vram_slowdown > options.maximum_slowdown))) {
      fail("interval slowdown exceeds the allowed percentage");
   }
   return summary;
}

std::size_t working_set(const sycl::device &device) {
   constexpr std::size_t cap = 256ull << 20;
   const auto memory =
      device.get_info<sycl::info::device::global_mem_size>();
   return std::max<std::size_t>(4096,
      (std::min<std::size_t>(cap, memory / 8) / 4096) * 4096);
}

struct OutputOptions {
   OutputFormat format = OutputFormat::text;
   std::string path;
};

struct SelectionOptions {
   std::vector<std::string> selectors;
   bool all = false;
   bool all_gpus = false;
};

struct BenchmarkOptions {
   SelectionOptions selection;
   OutputOptions output;
   std::set<std::string> tests{"all"};
};

class OutputTarget {
public:
   explicit OutputTarget(const OutputOptions &options) {
      if (options.path.empty() || options.path == "-") {
         stream_ = &std::cout;
      } else {
         file_.open(options.path);
         if (!file_) {
            throw std::runtime_error("unable to open output file: " +
               options.path);
         }
         stream_ = &file_;
      }
   }

   std::ostream &stream() { return *stream_; }

private:
   std::ofstream file_;
   std::ostream *stream_ = nullptr;
};

OutputFormat parse_output_format(const std::string &value) {
   if (value == "text") {
      return OutputFormat::text;
   }
   if (value == "json") {
      return OutputFormat::json;
   }
   if (value == "jsonl") {
      return OutputFormat::jsonl;
   }
   if (value == "csv") {
      return OutputFormat::csv;
   }
   throw std::invalid_argument("invalid output format: " + value +
      " (expected text, json, jsonl, or csv)");
}

std::chrono::seconds parse_duration(
   const std::string &text, const char *option) {
   std::string value = text;
   if (value.empty()) {
      throw std::invalid_argument(std::string(option) +
         " requires a positive duration");
   }
   std::uint64_t multiplier = 1;
   const char suffix = value.back();
   if (suffix < '0' || suffix > '9') {
      value.pop_back();
      if (suffix == 's') {
         multiplier = 1;
      } else if (suffix == 'm') {
         multiplier = 60;
      } else if (suffix == 'h') {
         multiplier = 3600;
      } else {
         throw std::invalid_argument(std::string("invalid duration for ") +
            option + ": " + text + " (use s, m, or h)");
      }
   }
   if (value.empty() ||
       !std::all_of(value.begin(), value.end(),
          [](char c) { return c >= '0' && c <= '9'; })) {
      throw std::invalid_argument(
         std::string("invalid duration for ") + option + ": " + text);
   }
   std::uint64_t amount = 0;
   try {
      amount = std::stoull(value);
   } catch (const std::exception &) {
      throw std::invalid_argument(
         std::string("invalid duration for ") + option + ": " + text);
   }
   constexpr std::uint64_t maximum_seconds = 10ull * 365 * 24 * 3600;
   if (amount == 0 || amount > maximum_seconds / multiplier) {
      throw std::invalid_argument(
         std::string("duration out of range for ") + option + ": " + text);
   }
   return std::chrono::seconds(amount * multiplier);
}

StressProfile parse_stress_profile(const std::string &value) {
   if (value == "compute") {
      return StressProfile::compute;
   }
   if (value == "vram") {
      return StressProfile::vram;
   }
   if (value == "mixed") {
      return StressProfile::mixed;
   }
   throw std::invalid_argument("invalid stress profile: " + value +
      " (expected compute, vram, or mixed)");
}

StressComputeWorkload parse_stress_compute_workload(
   const std::string &value) {
   if (value == "fp32") {
      return StressComputeWorkload::fp32;
   }
   if (value == "fp64") {
      return StressComputeWorkload::fp64;
   }
   if (value == "matrix-fp16") {
      return StressComputeWorkload::matrix_fp16;
   }
   if (value == "matrix-bf16") {
      return StressComputeWorkload::matrix_bf16;
   }
   if (value == "matrix-tf32") {
      return StressComputeWorkload::matrix_tf32;
   }
   if (value == "matrix-int8") {
      return StressComputeWorkload::matrix_int8;
   }
   if (value == "matrix-fp64") {
      return StressComputeWorkload::matrix_fp64;
   }
   throw std::invalid_argument("invalid compute workload: " + value +
      "; expected fp32, fp64, matrix-fp16, matrix-bf16, matrix-tf32, "
      "matrix-int8, or matrix-fp64");
}

unsigned parse_memory_percent(std::string value) {
   const std::string original = value;
   if (!value.empty() && value.back() == '%') {
      value.pop_back();
   }
   if (value.empty() ||
       !std::all_of(value.begin(), value.end(),
          [](char c) { return c >= '0' && c <= '9'; })) {
      throw std::invalid_argument("invalid memory percentage: " + original);
   }
   unsigned long result = 0;
   try {
      result = std::stoul(value);
   } catch (const std::exception &) {
      throw std::invalid_argument("invalid memory percentage: " + original);
   }
   if (result < 1 || result > 90) {
      throw std::out_of_range("memory percentage must be between 1 and 90");
   }
   return static_cast<unsigned>(result);
}

std::uint32_t parse_seed(const std::string &value) {
   std::size_t used = 0;
   unsigned long result = 0;
   try {
      result = std::stoul(value, &used, 0);
   } catch (const std::exception &) {
      throw std::invalid_argument("invalid seed: " + value);
   }
   if (used != value.size() ||
       result > std::numeric_limits<std::uint32_t>::max()) {
      throw std::invalid_argument("invalid seed: " + value);
   }
   return static_cast<std::uint32_t>(result);
}

double parse_nonnegative_double(
   const std::string &value, const char *option) {
   std::size_t used = 0;
   double result = 0.0;
   try {
      result = std::stod(value, &used);
   } catch (const std::exception &) {
      throw std::invalid_argument(std::string("invalid value for ") +
         option + ": " + value);
   }
   if (used != value.size() || !std::isfinite(result) || result < 0.0) {
      throw std::invalid_argument(std::string("invalid value for ") +
         option + ": " + value);
   }
   return result;
}

std::string lower_copy(std::string value) {
   std::transform(value.begin(), value.end(), value.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
   return value;
}

std::size_t parse_chunk_size(const std::string &text) {
   const auto suffix_begin = std::find_if(text.begin(), text.end(),
      [](char c) { return c < '0' || c > '9'; });
   const std::string amount_text(text.begin(), suffix_begin);
   const std::string suffix = lower_copy(
      std::string(suffix_begin, text.end()));
   if (amount_text.empty()) {
      throw std::invalid_argument("invalid chunk size: " + text);
   }
   std::uint64_t amount = 0;
   try {
      amount = std::stoull(amount_text);
   } catch (const std::exception &) {
      throw std::invalid_argument("invalid chunk size: " + text);
   }
   std::uint64_t multiplier = 0;
   if (suffix.empty() || suffix == "m" || suffix == "mb" ||
       suffix == "mib") {
      multiplier = 1ull << 20;
   } else if (suffix == "g" || suffix == "gb" || suffix == "gib") {
      multiplier = 1ull << 30;
   } else {
      throw std::invalid_argument("invalid chunk size: " + text +
         " (use MiB or GiB)");
   }
   if (amount == 0 ||
       amount > std::numeric_limits<std::size_t>::max() / multiplier) {
      throw std::out_of_range("chunk size out of range: " + text);
   }
   return static_cast<std::size_t>(amount * multiplier);
}

std::optional<std::string> physical_device_key(const sycl::device &device) {
   if (device.has(sycl::aspect::ext_intel_pci_address)) {
      try {
         const std::string address = device.get_info<
            sycl::ext::intel::info::device::pci_address>();
         if (!address.empty()) {
            return "pci:" + lower_copy(address);
         }
      } catch (const sycl::exception &) {
         // Fall back to the device UUID when PCI information is advertised
         // but cannot be queried through this backend.
      }
   }
   if (device.has(sycl::aspect::ext_intel_device_info_uuid)) {
      try {
         const auto uuid = device.get_info<
            sycl::ext::intel::info::device::uuid>();
         std::ostringstream key;
         key << "uuid:" << std::hex << std::setfill('0');
         for (const unsigned char byte : uuid) {
            key << std::setw(2) << static_cast<unsigned>(byte);
         }
         return key.str();
      } catch (const sycl::exception &) {
         // Without a backend-independent identifier, keep both devices.
      }
   }
   return std::nullopt;
}

bool same_physical_gpu(
   const sycl::device &left, const sycl::device &right) {
   const auto left_backend = left.get_backend();
   const auto right_backend = right.get_backend();
   const bool opencl_level_zero_pair =
      (left_backend == sycl::backend::opencl &&
         right_backend == sycl::backend::ext_oneapi_level_zero) ||
      (left_backend == sycl::backend::ext_oneapi_level_zero &&
         right_backend == sycl::backend::opencl);
   if (!opencl_level_zero_pair) {
      return false;
   }
   const auto left_key = physical_device_key(left);
   const auto right_key = physical_device_key(right);
   return left_key && right_key && *left_key == *right_key;
}

std::vector<DeviceEntry> select_devices(
   const std::vector<DeviceEntry> &devices,
   const SelectionOptions &selection) {
   if (selection.all && selection.all_gpus) {
      throw std::invalid_argument("--all and --all-gpus are mutually exclusive");
   }
   if ((selection.all || selection.all_gpus) &&
       !selection.selectors.empty()) {
      throw std::invalid_argument(
         "--device cannot be combined with --all or --all-gpus");
   }
   std::vector<DeviceEntry> selected;
   if (selection.all || selection.all_gpus) {
      for (const auto &entry : devices) {
         if (selection.all) {
            selected.push_back(entry);
         } else if (entry.device.is_gpu()) {
            auto duplicate = std::find_if(selected.begin(), selected.end(),
               [&](const DeviceEntry &existing) {
                  return same_physical_gpu(existing.device, entry.device);
               });
            if (duplicate == selected.end()) {
               selected.push_back(entry);
            } else if (duplicate->device.get_backend() ==
                          sycl::backend::opencl &&
                       entry.device.get_backend() ==
                          sycl::backend::ext_oneapi_level_zero) {
               *duplicate = entry;
            }
         }
      }
   } else {
      for (const std::string &selector : selection.selectors) {
         std::vector<DeviceEntry> matches;
         const bool numeric = !selector.empty() &&
            std::all_of(selector.begin(), selector.end(),
               [](char c) { return c >= '0' && c <= '9'; });
         for (const auto &entry : devices) {
            const std::string name = entry.device.get_info<
               sycl::info::device::name>();
            if ((numeric && std::to_string(entry.id) == selector) ||
                (!numeric && (entry.selector == selector ||
                   lower_copy(name).find(lower_copy(selector)) !=
                      std::string::npos))) {
               matches.push_back(entry);
            }
         }
         if (matches.empty()) {
            throw std::out_of_range("no device matches selector: " + selector);
         }
         if (matches.size() != 1) {
            throw std::invalid_argument("ambiguous device selector: " +
               selector + " (use backend:index from the devices command)");
         }
         if (std::none_of(selected.begin(), selected.end(),
                [&](const DeviceEntry &entry) {
                   return entry.id == matches.front().id;
                })) {
            selected.push_back(matches.front());
         }
      }
   }
   if (selected.empty()) {
      throw std::invalid_argument(
         "select a device with --device, --all, or --all-gpus");
   }
   return selected;
}

bool matrix_capable(const sycl::device &device) {
   return supports_any_matrix_variant<SpirvFp16WideCase,
      SpirvFp16NarrowCase, SpirvCpuFp16Case, SpirvBf16WideCase,
      SpirvBf16NarrowCase, SpirvCpuBf16Case, SpirvTf32WideCase,
      SpirvCpuTf32Case, SpirvInt8WideCase, SpirvInt8NarrowCase,
      SpirvCpuInt8Case, Fp16Case, Bf16Case, Tf32Case, Int8Case,
      Fp64Case, Amd90aFp16Case, Amd90aBf16Case, Amd90aInt8Case,
      Amd90aFp64Case>(device);
}

void write_device_json(std::ostream &out, const DeviceEntry &entry) {
   const auto &device = entry.device;
   out << "{\"id\":" << entry.id
       << ",\"selector\":\"" << json_escape(entry.selector)
       << "\",\"backend\":\"" << backend_name(device.get_backend())
       << "\",\"type\":\"" << device_type_name(device)
       << "\",\"name\":\"" << json_escape(device.get_info<
             sycl::info::device::name>())
       << "\",\"vendor\":\"" << json_escape(device.get_info<
             sycl::info::device::vendor>())
       << "\",\"driver\":\"" << json_escape(device.get_info<
             sycl::info::device::driver_version>())
       << "\",\"platform\":\"" << json_escape(device.get_platform().get_info<
             sycl::info::platform::name>())
       << "\",\"memory_bytes\":" << device.get_info<
             sycl::info::device::global_mem_size>()
       << ",\"local_memory_bytes\":" << device.get_info<
             sycl::info::device::local_mem_size>()
       << ",\"compute_units\":" << device.get_info<
             sycl::info::device::max_compute_units>()
       << ",\"max_clock_mhz\":" << device.get_info<
             sycl::info::device::max_clock_frequency>()
       << ",\"usm_device\":"
       << (device.has(sycl::aspect::usm_device_allocations) ? "true" : "false")
       << ",\"fp64\":" << (device.has(sycl::aspect::fp64) ? "true" : "false")
       << ",\"matrix\":" << (matrix_capable(device) ? "true" : "false")
       << '}';
}

void render_devices(const std::vector<DeviceEntry> &devices,
   OutputFormat format, std::ostream &out) {
   if (format == OutputFormat::text) {
      out << "Available SYCL devices\n"
          << "  ID  Selector       Type   Memory MiB  USM  Matrix  Name\n";
      for (const auto &entry : devices) {
         const auto memory_mib = entry.device.get_info<
            sycl::info::device::global_mem_size>() / (1024 * 1024);
         out << "  " << std::left << std::setw(3) << entry.id
             << ' ' << std::setw(14) << entry.selector
             << ' ' << std::setw(6) << device_type_name(entry.device)
             << ' ' << std::right << std::setw(10) << memory_mib << "  "
             << (entry.device.has(sycl::aspect::usm_device_allocations)
                    ? "yes" : " no")
             << "  " << std::setw(6)
             << (matrix_capable(entry.device) ? "yes" : "no") << "  "
             << entry.device.get_info<sycl::info::device::name>() << '\n';
      }
      return;
   }
   if (format == OutputFormat::csv) {
      out << "id,selector,backend,type,name,vendor,driver,platform,memory_bytes,"
             "local_memory_bytes,compute_units,max_clock_mhz,usm_device,fp64,"
             "matrix\n";
      for (const auto &entry : devices) {
         const auto &device = entry.device;
         out << entry.id << ',' << csv_escape(entry.selector) << ','
             << backend_name(device.get_backend()) << ','
             << device_type_name(device) << ','
             << csv_escape(device.get_info<sycl::info::device::name>()) << ','
             << csv_escape(device.get_info<sycl::info::device::vendor>()) << ','
             << csv_escape(device.get_info<sycl::info::device::driver_version>())
             << ',' << csv_escape(device.get_platform().get_info<
                   sycl::info::platform::name>())
             << ',' << device.get_info<sycl::info::device::global_mem_size>()
             << ',' << device.get_info<sycl::info::device::local_mem_size>()
             << ',' << device.get_info<sycl::info::device::max_compute_units>()
             << ',' << device.get_info<sycl::info::device::max_clock_frequency>()
             << ',' << device.has(sycl::aspect::usm_device_allocations)
             << ',' << device.has(sycl::aspect::fp64)
             << ',' << matrix_capable(device) << '\n';
      }
      return;
   }
   if (format == OutputFormat::json) {
      out << '[';
   }
   for (std::size_t i = 0; i < devices.size(); ++i) {
      if (format == OutputFormat::json && i != 0) {
         out << ',';
      }
      write_device_json(out, devices[i]);
      if (format == OutputFormat::jsonl) {
         out << '\n';
      }
   }
   if (format == OutputFormat::json) {
      out << "]\n";
   }
}

const std::set<std::string> all_benchmark_tests{
   "fp64", "fp32", "matrix-fp16", "matrix-bf16", "matrix-tf32",
   "matrix-int8", "matrix-fp64", "memory", "transfer"};

std::set<std::string> parse_tests(const std::string &value) {
   if (value == "all") {
      return {"all"};
   }
   std::set<std::string> result;
   std::size_t begin = 0;
   while (begin <= value.size()) {
      const std::size_t end = value.find(',', begin);
      const std::string name = value.substr(begin,
         end == std::string::npos ? std::string::npos : end - begin);
      if (all_benchmark_tests.count(name) == 0) {
         throw std::invalid_argument("unknown benchmark test: " + name);
      }
      result.insert(name);
      if (end == std::string::npos) {
         break;
      }
      begin = end + 1;
   }
   return result;
}

bool test_enabled(const std::set<std::string> &tests,
   const char *name) {
   return tests.count("all") != 0 || tests.count(name) != 0;
}

struct BenchmarkReport {
   DeviceEntry entry;
   std::string timestamp;
   std::string status = "pass";
   std::string message;
   std::size_t sample_bytes = 0;
   std::vector<BenchmarkMeasurement> measurements;
};

BenchmarkReport run_benchmark(const DeviceEntry &entry,
   const BenchmarkOptions &options) {
   const auto &device = entry.device;
   if (!device.has(sycl::aspect::usm_device_allocations)) {
      throw std::runtime_error("device USM is required");
   }
   sycl::queue q(device);
   BenchmarkReport report{entry};
   report.timestamp = utc_timestamp();
   report.sample_bytes = working_set(device);
   if (test_enabled(options.tests, "fp64")) {
      if (device.has(sycl::aspect::fp64)) {
         report.measurements.push_back(
            benchmark_compute<double>(q, device, "FP64"));
      } else {
         BenchmarkMeasurement result{"arithmetic", "FP64"};
         result.status = "unavailable";
         result.message = "fp64 is not supported";
         report.measurements.push_back(std::move(result));
      }
   }
   if (test_enabled(options.tests, "fp32")) {
      report.measurements.push_back(
         benchmark_compute<float>(q, device, "FP32"));
   }
   if (test_enabled(options.tests, "matrix-fp16")) {
      append_matrix_variants<SpirvFp16WideCase, SpirvFp16NarrowCase,
         SpirvCpuFp16Case, Fp16Case, Amd90aFp16Case>(
         report.measurements, q, device, "FP16->FP32");
   }
   if (test_enabled(options.tests, "matrix-bf16")) {
      append_matrix_variants<SpirvBf16WideCase, SpirvBf16NarrowCase,
         SpirvCpuBf16Case, Bf16Case, Amd90aBf16Case>(
         report.measurements, q, device, "BF16->FP32");
   }
   if (test_enabled(options.tests, "matrix-tf32")) {
      append_matrix_variants<SpirvTf32WideCase, SpirvCpuTf32Case,
         Tf32Case>(report.measurements, q, device, "TF32->FP32");
   }
   if (test_enabled(options.tests, "matrix-int8")) {
      append_matrix_variants<SpirvInt8WideCase, SpirvInt8NarrowCase,
         SpirvCpuInt8Case, Int8Case, Amd90aInt8Case>(
         report.measurements, q, device, "INT8->INT32");
   }
   if (test_enabled(options.tests, "matrix-fp64")) {
      append_matrix_variants<Fp64Case, Amd90aFp64Case>(
         report.measurements, q, device, "FP64->FP64");
   }
   if (test_enabled(options.tests, "memory")) {
      try {
         const auto rates = memory_bandwidth(q, device, report.sample_bytes);
         report.measurements.push_back(
            {"memory", "read", "ok", rates.read, "GB/s"});
         report.measurements.push_back(
            {"memory", "write", "ok", rates.write, "GB/s"});
      } catch (const std::exception &e) {
         BenchmarkMeasurement result{"memory", "read-write"};
         result.status = "error";
         result.message = concise_error(e);
         report.measurements.push_back(std::move(result));
         report.status = "error";
      }
   }
   if (test_enabled(options.tests, "transfer")) {
      if (!device.has(sycl::aspect::usm_host_allocations)) {
         BenchmarkMeasurement result{"transfer", "all"};
         result.status = "unavailable";
         result.message = "host USM allocation is not supported";
         report.measurements.push_back(std::move(result));
      } else {
         try {
            const auto rates =
               transfer_bandwidth(q, device, report.sample_bytes);
            report.measurements.push_back({"transfer", "host-to-device",
               "ok", rates.host_to_device, "GB/s"});
            report.measurements.push_back({"transfer", "device-to-host",
               "ok", rates.device_to_host, "GB/s"});
            report.measurements.push_back({"transfer", "concurrent-both",
               "ok", rates.bidirectional, "GB/s"});
         } catch (const std::exception &e) {
            BenchmarkMeasurement result{"transfer", "all"};
            result.status = "error";
            result.message = concise_error(e);
            report.measurements.push_back(std::move(result));
            report.status = "error";
         }
      }
   }
   for (const auto &measurement : report.measurements) {
      if (measurement.status == "error") {
         report.status = "error";
         if (report.message.empty()) {
            report.message = measurement.name + ": " + measurement.message;
         }
      }
   }
   return report;
}

void write_measurement_json(std::ostream &out,
   const BenchmarkMeasurement &measurement) {
   out << "{\"category\":\"" << json_escape(measurement.category)
       << "\",\"name\":\"" << json_escape(measurement.name)
       << "\",\"status\":\"" << measurement.status << '"';
   if (measurement.status == "ok") {
      out << ",\"value\":" << std::setprecision(12) << measurement.value
          << ",\"unit\":\"" << json_escape(measurement.unit) << '"';
   }
   if (!measurement.detail.empty()) {
      out << ",\"detail\":\"" << json_escape(measurement.detail) << '"';
   }
   if (!measurement.message.empty()) {
      out << ",\"message\":\"" << json_escape(measurement.message) << '"';
   }
   out << '}';
}

void write_benchmark_json(std::ostream &out,
   const BenchmarkReport &report) {
   out << "{\"schema_version\":1,\"tool_version\":\""
       << tool_version << "\",\"timestamp\":\""
       << json_escape(report.timestamp) << "\",\"device\":";
   write_device_json(out, report.entry);
   out << ",\"status\":\"" << report.status
       << "\",\"sample_bytes\":" << report.sample_bytes;
   if (!report.message.empty()) {
      out << ",\"message\":\"" << json_escape(report.message) << '"';
   }
   out << ",\"measurements\":[";
   for (std::size_t i = 0; i < report.measurements.size(); ++i) {
      if (i != 0) {
         out << ',';
      }
      write_measurement_json(out, report.measurements[i]);
   }
   out << "]}";
}

const char *category_title(const std::string &category) {
   if (category == "arithmetic") {
      return "Arithmetic throughput";
   }
   if (category == "matrix") {
      return "Matrix throughput";
   }
   if (category == "memory") {
      return "Device-memory throughput";
   }
   return "USM transfer throughput";
}

void render_benchmarks(const std::vector<BenchmarkReport> &reports,
   OutputFormat format, std::ostream &out) {
   if (format == OutputFormat::text) {
      for (const auto &report : reports) {
         const auto &device = report.entry.device;
         out << "\nSYCL throughput report\n"
             << "  timestamp        : " << report.timestamp << '\n'
             << "  device           : " << report.entry.selector << " (id "
             << report.entry.id << ")\n"
             << "  name             : "
             << device.get_info<sycl::info::device::name>() << '\n'
             << "  vendor           : "
             << device.get_info<sycl::info::device::vendor>() << '\n'
             << "  driver           : "
             << device.get_info<sycl::info::device::driver_version>() << '\n'
             << "  execution.units  : "
             << device.get_info<sycl::info::device::max_compute_units>()
             << '\n' << "  reported.clock   : "
             << device.get_info<sycl::info::device::max_clock_frequency>()
             << " MHz (static)\n"
             << "  memory.capacity  : "
             << device.get_info<sycl::info::device::global_mem_size>() /
                   (1024 * 1024)
             << " MiB global, " << device.get_info<
                   sycl::info::device::local_mem_size>() / 1024
             << " KiB local\n  sample.bytes     : " << report.sample_bytes
             << '\n';
         if (report.status == "error") {
            out << "  result           : ERROR " << report.message << '\n';
         }
         std::string category;
         for (const auto &measurement : report.measurements) {
            if (measurement.category != category) {
               category = measurement.category;
               out << '\n' << category_title(category) << '\n';
            }
            out << "  " << std::left << std::setw(22) << measurement.name;
            if (measurement.status == "ok") {
               out << std::right << std::setw(10) << std::fixed
                   << std::setprecision(3) << measurement.value << ' '
                   << measurement.unit;
               if (!measurement.detail.empty()) {
                  out << "  (" << measurement.detail << ')';
               }
               out << '\n';
            } else {
               out << measurement.status << ": " << measurement.message
                   << '\n';
            }
         }
      }
      return;
   }
   if (format == OutputFormat::json) {
      out << '[';
      for (std::size_t i = 0; i < reports.size(); ++i) {
         if (i != 0) {
            out << ',';
         }
         write_benchmark_json(out, reports[i]);
      }
      out << "]\n";
      return;
   }
   if (format == OutputFormat::jsonl) {
      for (const auto &report : reports) {
         write_benchmark_json(out, report);
         out << '\n';
      }
      return;
   }
   out << "timestamp,selector,device_name,report_status,category,test,test_status,"
          "value,unit,detail,message\n";
   for (const auto &report : reports) {
      for (const auto &measurement : report.measurements) {
         out << csv_escape(report.timestamp) << ','
             << csv_escape(report.entry.selector) << ','
             << csv_escape(report.entry.device.get_info<
                   sycl::info::device::name>()) << ','
             << report.status << ',' << csv_escape(measurement.category)
             << ',' << csv_escape(measurement.name) << ','
             << measurement.status << ',';
         if (measurement.status == "ok") {
            out << std::setprecision(12) << measurement.value;
         }
         out << ',' << csv_escape(measurement.unit) << ','
             << csv_escape(measurement.detail) << ','
             << csv_escape(measurement.message) << '\n';
      }
   }
}

double interval_average(double sum, std::size_t count) {
   return count == 0 ? 0.0 : sum / count;
}

void write_rate_summary_json(std::ostream &out,
   const RateSummary &summary, const char *unit) {
   out << "{\"samples\":" << summary.samples
       << ",\"average\":" << std::setprecision(12) << summary.average
       << ",\"minimum\":" << summary.minimum
       << ",\"p95\":" << summary.p95
       << ",\"unit\":\"" << unit << "\"}";
}

void write_stress_summary_json(std::ostream &out,
   const StressSummary &summary) {
   out << "{\"schema_version\":2,\"tool_version\":\""
       << tool_version << "\",\"timestamp\":\""
       << json_escape(summary.timestamp) << "\",\"device\":";
   write_device_json(out, summary.entry);
   out << ",\"profile\":\"" << summary.profile
       << "\",\"compute_workload\":\"" << summary.compute_workload
       << "\",\"compute_unit\":\"" << summary.compute_unit
       << "\",\"requested_seconds\":" << summary.requested_seconds
       << ",\"memory_percent\":" << summary.memory_percent
       << ",\"seed\":" << summary.seed
       << ",\"status\":\"" << summary.status
       << "\",\"elapsed_seconds\":" << std::setprecision(12)
       << summary.elapsed_seconds << ",\"memory_bytes\":"
       << summary.memory_bytes << ",\"memory_chunks\":"
       << summary.memory_chunks << ",\"memory_max_chunk_bytes\":"
       << summary.memory_max_chunk_bytes
       << ",\"compute_slowdown_percent\":"
       << summary.compute_slowdown << ",\"vram_slowdown_percent\":"
       << summary.vram_slowdown << ",\"compute\":";
   write_rate_summary_json(out, summary.compute, summary.compute_unit.c_str());
   out << ",\"vram\":";
   write_rate_summary_json(out, summary.vram, "GB/s");
   if (!summary.message.empty()) {
      out << ",\"message\":\"" << json_escape(summary.message) << '"';
   }
   out << '}';
}

class StressRenderer {
public:
   StressRenderer(OutputFormat format, std::ostream &out,
      const StressOptions &options,
      const std::vector<DeviceEntry> &devices)
      : format_(format), out_(out),
        compute_workload_(stress_compute_workload_name(
           options.compute_workload)),
        compute_unit_(stress_compute_unit(options.compute_workload)) {
      if (format_ == OutputFormat::text) {
         out_ << "SYCL stress plan\n"
              << "  profile          : "
              << stress_profile_name(options.profile) << '\n'
              << "  duration         : " << options.duration.count()
              << " seconds\n"
              << "  report.interval  : "
              << options.report_interval.count() << " seconds\n";
         if (options.profile != StressProfile::vram) {
            out_ << "  compute.workload : " << compute_workload_ << '\n';
         }
         if (options.profile != StressProfile::compute) {
            out_ << "  memory.request   : " << options.memory_percent
                 << "%\n"
                 << "  memory.chunk     : up to "
                 << options.chunk_size_bytes / (1024 * 1024)
                 << " MiB (also capped by device limit)\n";
         }
         out_ << "  execution        : "
              << (options.parallel ? "parallel" : "sequential") << '\n'
              << "  seed             : 0x" << std::hex << options.seed
              << std::dec << "\n  devices          :\n";
         for (const auto &entry : devices) {
            out_ << "    " << entry.selector << "  " << entry.device.get_info<
               sycl::info::device::name>() << '\n';
         }
         out_ << std::flush;
      } else if (format_ == OutputFormat::csv) {
         out_ << "timestamp,event,selector,elapsed_seconds,status,"
                 "compute_workload,compute_unit,memory_bytes,"
                 "memory_chunks,memory_max_chunk_bytes,compute_samples,"
                 "compute_average,compute_minimum,"
                 "compute_slowdown_percent,vram_samples,vram_average_gbps,"
                 "vram_minimum_gbps,vram_slowdown_percent,message\n";
      }
   }

   void progress(const DeviceEntry &entry, Clock::duration elapsed,
      const StressInterval &interval) {
      std::lock_guard<std::mutex> lock(mutex_);
      const double seconds = std::chrono::duration<double>(elapsed).count();
      const double compute_average = interval_average(
         interval.compute_sum, interval.compute_batches);
      const double vram_average = interval_average(
         interval.vram_sum, interval.vram_passes);
      if (format_ == OutputFormat::text) {
         out_ << "  [" << entry.selector << " | " << entry.device.get_info<
            sycl::info::device::name>() << " | " << stress_elapsed(elapsed)
              << ']';
         if (interval.compute_batches != 0) {
            out_ << " compute=" << std::fixed << std::setprecision(3)
                 << compute_average << ' ' << compute_unit_ << " (min "
                 << interval.compute_min << ')';
         }
         if (interval.vram_passes != 0) {
            out_ << " vram=" << std::fixed << std::setprecision(3)
                 << vram_average << " GB/s (min " << interval.vram_min
                 << ')';
         }
         out_ << " errors=0\n" << std::flush;
      } else if (format_ == OutputFormat::jsonl) {
         out_ << "{\"schema_version\":2,\"tool_version\":\""
              << tool_version << "\",\"timestamp\":\""
              << utc_timestamp()
              << "\",\"event\":\"interval\",\"selector\":\""
              << json_escape(entry.selector) << "\",\"elapsed_seconds\":"
              << seconds << ",\"compute_workload\":\""
              << compute_workload_ << "\",\"compute_unit\":\""
              << compute_unit_ << "\",\"compute_samples\":"
              << interval.compute_batches << ",\"compute_average\":"
              << compute_average << ",\"compute_minimum\":"
              << (interval.compute_batches ? interval.compute_min : 0.0)
              << ",\"vram_samples\":" << interval.vram_passes
              << ",\"vram_average_gbps\":" << vram_average
              << ",\"vram_minimum_gbps\":"
              << (interval.vram_passes ? interval.vram_min : 0.0)
              << ",\"errors\":0}\n" << std::flush;
      } else if (format_ == OutputFormat::csv) {
         out_ << csv_escape(utc_timestamp()) << ",interval,"
              << csv_escape(entry.selector) << ','
              << seconds << ",running," << compute_workload_ << ','
              << compute_unit_ << ",0,0,0," << interval.compute_batches
              << ','
              << compute_average << ','
              << (interval.compute_batches ? interval.compute_min : 0.0)
              << ",0," << interval.vram_passes << ',' << vram_average << ','
              << (interval.vram_passes ? interval.vram_min : 0.0)
              << ",0,\n" << std::flush;
      }
   }

   void finish(const std::vector<StressSummary> &summaries) {
      std::lock_guard<std::mutex> lock(mutex_);
      if (format_ == OutputFormat::json) {
         out_ << '[';
         for (std::size_t i = 0; i < summaries.size(); ++i) {
            if (i != 0) {
               out_ << ',';
            }
            write_stress_summary_json(out_, summaries[i]);
         }
         out_ << "]\n";
         return;
      }
      for (const auto &summary : summaries) {
         if (format_ == OutputFormat::text) {
            out_ << "\nStress summary: " << summary.entry.selector << " ("
                 << summary.entry.device.get_info<sycl::info::device::name>()
                 << ")\n"
                 << "  result           : " << summary.status << '\n'
                 << "  elapsed          : " << std::fixed
                 << std::setprecision(3) << summary.elapsed_seconds
                 << " seconds\n";
            if (summary.memory_bytes != 0) {
               out_ << "  memory.tested    : "
                    << summary.memory_bytes / (1024 * 1024) << " MiB\n"
                    << "  memory.chunks    : " << summary.memory_chunks
                    << " (up to "
                    << summary.memory_max_chunk_bytes / (1024 * 1024)
                    << " MiB each)\n";
            }
            if (summary.compute.samples) {
               out_ << "  compute          : " << summary.compute_workload
                    << ", avg " << summary.compute.average
                    << ", min " << summary.compute.minimum << ", p95 "
                    << summary.compute.p95 << ' ' << summary.compute_unit << " ("
                    << summary.compute.samples << " batches)\n"
                    << "  compute.slowdown : " << summary.compute_slowdown
                    << "%\n";
            }
            if (summary.vram.samples) {
               out_ << "  vram             : avg " << summary.vram.average
                    << ", min " << summary.vram.minimum << ", p95 "
                    << summary.vram.p95 << " GB/s (" << summary.vram.samples
                    << " passes)\n"
                    << "  vram.slowdown    : " << summary.vram_slowdown
                    << "%\n";
            }
            if (!summary.message.empty()) {
               out_ << "  message          : " << summary.message << '\n';
            }
         } else if (format_ == OutputFormat::jsonl) {
            out_ << "{\"event\":\"summary\",\"summary\":";
            write_stress_summary_json(out_, summary);
            out_ << "}\n";
         } else {
            out_ << csv_escape(summary.timestamp) << ",summary,"
                 << csv_escape(summary.entry.selector) << ','
                 << summary.elapsed_seconds << ',' << summary.status << ','
                 << summary.compute_workload << ',' << summary.compute_unit
                 << ','
                 << summary.memory_bytes << ',' << summary.memory_chunks << ','
                 << summary.memory_max_chunk_bytes << ','
                 << summary.compute.samples << ',' << summary.compute.average
                 << ',' << summary.compute.minimum << ','
                 << summary.compute_slowdown << ','
                 << summary.vram.samples << ',' << summary.vram.average << ','
                 << summary.vram.minimum << ',' << summary.vram_slowdown << ','
                 << csv_escape(summary.message) << '\n';
         }
      }
      out_ << std::flush;
   }

private:
   OutputFormat format_;
   std::ostream &out_;
   std::string compute_workload_;
   std::string compute_unit_;
   std::mutex mutex_;
};

bool parse_common_output(const std::string &arg, int &i, int argc,
   char **argv, OutputOptions &output) {
   const auto require_value = [&](const char *option) -> std::string {
      if (i + 1 >= argc) {
         throw std::invalid_argument(std::string(option) +
            " requires a value");
      }
      return argv[++i];
   };
   if (arg == "--format") {
      output.format = parse_output_format(require_value("--format"));
      return true;
   }
   if (arg == "--output") {
      output.path = require_value("--output");
      return true;
   }
   return false;
}

bool parse_selection_option(const std::string &arg, int &i, int argc,
   char **argv, SelectionOptions &selection, bool allow_all_gpus) {
   if (arg == "--device") {
      if (i + 1 >= argc) {
         throw std::invalid_argument("--device requires a value");
      }
      selection.selectors.push_back(argv[++i]);
      return true;
   }
   if (!allow_all_gpus && arg == "--all") {
      selection.all = true;
      return true;
   }
   if (allow_all_gpus && arg == "--all-gpus") {
      selection.all_gpus = true;
      return true;
   }
   return false;
}

void show_main_help(const char *program) {
   std::cout
      << "SyclBenchmark 1.0\n\n"
      << "Usage: " << program << " <command> [options]\n\n"
      << "Commands:\n"
      << "  devices    list devices and stable backend selectors\n"
      << "  benchmark  measure selected throughput tests\n"
      << "  stress     run bounded compute and VRAM validation\n\n"
      << "Run '" << program << " <command> --help' for command options.\n";
}

void show_devices_help(const char *program) {
   std::cout << "Usage: " << program
             << " devices [--format text|json|jsonl|csv] [--output FILE]\n";
}

void show_benchmark_help(const char *program) {
   std::cout
      << "Usage: " << program
      << " benchmark (--device SELECTOR ... | --all) [options]\n\n"
      << "Options:\n"
      << "  --device SELECTOR  global ID, backend:index, or unique name text\n"
      << "  --all              benchmark every visible SYCL device\n"
      << "  --tests LIST       comma-separated tests (default: all)\n"
      << "  --format FORMAT    text, json, jsonl, or csv\n"
      << "  --output FILE      write output to FILE; '-' means stdout\n\n"
      << "Tests: fp64, fp32, matrix-fp16, matrix-bf16, matrix-tf32,\n"
      << "       matrix-int8, matrix-fp64, memory, transfer\n";
}

void show_stress_help(const char *program) {
   std::cout
      << "Usage: " << program
      << " stress --duration TIME (--device SELECTOR ... | --all-gpus) [options]\n\n"
      << "Options:\n"
      << "  --duration TIME        positive duration with s, m, or h suffix\n"
      << "  --device SELECTOR      global ID, backend:index, or unique name text\n"
      << "  --all-gpus             select every visible GPU\n"
      << "  --profile PROFILE      compute, vram, or mixed (default: mixed)\n"
      << "  --compute-workload W   fp32, fp64, matrix-fp16, matrix-bf16,\n"
      << "                         matrix-tf32, matrix-int8, or matrix-fp64\n"
      << "                         (default: fp32)\n"
      << "  --memory PERCENT       device memory to validate, 1-90 (default: 50)\n"
      << "  --chunk-size SIZE      maximum allocation chunk in MiB or GiB\n"
      << "                         (default: 512MiB; bare numbers mean MiB)\n"
      << "  --report-interval TIME progress interval (default: 5s)\n"
      << "  --seed VALUE           decimal or 0x-prefixed pattern seed\n"
      << "  --min-compute-rate N   fail below selected workload's average rate\n"
      << "  --min-vram-rate N      fail below N GB/s average\n"
      << "  --max-slowdown N       fail above N percent interval slowdown\n"
      << "  --parallel             run selected devices together (default)\n"
      << "  --sequential           run one selected device at a time\n"
      << "  --format FORMAT        text, json, jsonl, or csv\n"
      << "  --output FILE          write output to FILE; '-' means stdout\n";
}

int run_devices_command(int argc, char **argv,
   const std::vector<DeviceEntry> &devices) {
   OutputOptions output;
   for (int i = 2; i < argc; ++i) {
      const std::string arg(argv[i]);
      if (arg == "--help") {
         show_devices_help(argv[0]);
         return 0;
      }
      if (!parse_common_output(arg, i, argc, argv, output)) {
         throw std::invalid_argument("unknown devices option: " + arg);
      }
   }
   OutputTarget target(output);
   render_devices(devices, output.format, target.stream());
   return 0;
}

int run_benchmark_command(int argc, char **argv,
   const std::vector<DeviceEntry> &devices) {
   BenchmarkOptions options;
   for (int i = 2; i < argc; ++i) {
      const std::string arg(argv[i]);
      if (arg == "--help") {
         show_benchmark_help(argv[0]);
         return 0;
      }
      if (parse_common_output(arg, i, argc, argv, options.output) ||
          parse_selection_option(
             arg, i, argc, argv, options.selection, false)) {
         continue;
      }
      if (arg == "--tests") {
         if (i + 1 >= argc) {
            throw std::invalid_argument("--tests requires a value");
         }
         options.tests = parse_tests(argv[++i]);
      } else {
         throw std::invalid_argument("unknown benchmark option: " + arg);
      }
   }
   const auto selected = select_devices(devices, options.selection);
   std::vector<BenchmarkReport> reports;
   bool failed = false;
   for (const auto &entry : selected) {
      try {
         reports.push_back(run_benchmark(entry, options));
         failed = failed || reports.back().status == "error";
      } catch (const std::exception &e) {
         BenchmarkReport report{entry};
         report.timestamp = utc_timestamp();
         report.status = "error";
         report.message = concise_error(e);
         reports.push_back(std::move(report));
         failed = true;
      }
   }
   OutputTarget target(options.output);
   render_benchmarks(reports, options.output.format, target.stream());
   return failed ? 1 : 0;
}

int run_stress_command(int argc, char **argv,
   const std::vector<DeviceEntry> &devices) {
   StressOptions options;
   SelectionOptions selection;
   OutputOptions output;
   bool duration_seen = false;
   bool compute_workload_seen = false;
   for (int i = 2; i < argc; ++i) {
      const std::string arg(argv[i]);
      const auto require_value = [&](const char *option) -> std::string {
         if (i + 1 >= argc) {
            throw std::invalid_argument(std::string(option) +
               " requires a value");
         }
         return argv[++i];
      };
      if (arg == "--help") {
         show_stress_help(argv[0]);
         return 0;
      }
      if (parse_common_output(arg, i, argc, argv, output) ||
          parse_selection_option(arg, i, argc, argv, selection, true)) {
         continue;
      }
      if (arg == "--duration") {
         options.duration = parse_duration(
            require_value("--duration"), "--duration");
         duration_seen = true;
      } else if (arg == "--profile") {
         options.profile = parse_stress_profile(require_value("--profile"));
      } else if (arg == "--compute-workload") {
         options.compute_workload = parse_stress_compute_workload(
            require_value("--compute-workload"));
         compute_workload_seen = true;
      } else if (arg == "--memory") {
         options.memory_percent =
            parse_memory_percent(require_value("--memory"));
      } else if (arg == "--chunk-size") {
         options.chunk_size_bytes =
            parse_chunk_size(require_value("--chunk-size"));
      } else if (arg == "--report-interval") {
         options.report_interval = parse_duration(
            require_value("--report-interval"), "--report-interval");
      } else if (arg == "--seed") {
         options.seed = parse_seed(require_value("--seed"));
      } else if (arg == "--min-compute-rate") {
         options.minimum_compute_rate = parse_nonnegative_double(
            require_value("--min-compute-rate"), "--min-compute-rate");
      } else if (arg == "--min-vram-rate") {
         options.minimum_vram_rate = parse_nonnegative_double(
            require_value("--min-vram-rate"), "--min-vram-rate");
      } else if (arg == "--max-slowdown") {
         options.maximum_slowdown = parse_nonnegative_double(
            require_value("--max-slowdown"), "--max-slowdown");
      } else if (arg == "--parallel") {
         options.parallel = true;
      } else if (arg == "--sequential") {
         options.parallel = false;
      } else {
         throw std::invalid_argument("unknown stress option: " + arg);
      }
   }
   if (!duration_seen) {
      throw std::invalid_argument("stress requires --duration TIME");
   }
   if (options.profile == StressProfile::vram && compute_workload_seen) {
      throw std::invalid_argument(
         "--compute-workload requires compute or mixed profile");
   }
   if (options.profile == StressProfile::vram &&
       options.minimum_compute_rate >= 0.0) {
      throw std::invalid_argument(
         "--min-compute-rate requires compute or mixed profile");
   }
   if (options.profile == StressProfile::compute &&
       options.minimum_vram_rate >= 0.0) {
      throw std::invalid_argument(
         "--min-vram-rate requires vram or mixed profile");
   }
   const auto selected = select_devices(devices, selection);
   stress_stop_requested = 0;
   std::signal(SIGINT, request_stress_stop);
   std::signal(SIGTERM, request_stress_stop);
   OutputTarget target(output);
   StressRenderer renderer(output.format, target.stream(), options, selected);
   std::vector<StressSummary> summaries;
   summaries.reserve(selected.size());
   for (const auto &entry : selected) {
      StressSummary summary = make_stress_summary(entry, options);
      summary.status = "pending";
      summaries.push_back(std::move(summary));
   }
   std::atomic<bool> abort_requested{false};
   std::atomic<bool> failed{false};
   const StressProgress progress = [&](const DeviceEntry &entry,
      Clock::duration elapsed, const StressInterval &interval) {
      renderer.progress(entry, elapsed, interval);
   };
   const auto worker = [&](std::size_t index) {
      try {
         summaries[index] = stress_device(
            selected[index], options, progress, abort_requested);
         if (summaries[index].status == "fail") {
            failed.store(true);
            abort_requested.store(true);
         }
      } catch (const std::exception &e) {
         summaries[index] = make_stress_summary(selected[index], options);
         summaries[index].status = "error";
         summaries[index].message = concise_error(e);
         failed.store(true);
         abort_requested.store(true);
      }
   };
   if (options.parallel && selected.size() > 1) {
      std::vector<std::thread> threads;
      for (std::size_t i = 0; i < selected.size(); ++i) {
         threads.emplace_back(worker, i);
      }
      for (auto &thread : threads) {
         thread.join();
      }
   } else {
      for (std::size_t i = 0; i < selected.size(); ++i) {
         if (abort_requested.load() || stress_stop_requested) {
            summaries[i].status = stress_stop_requested
               ? "interrupted" : "skipped";
            continue;
         }
         worker(i);
      }
   }
   renderer.finish(summaries);
   if (stress_stop_requested) {
      return 130;
   }
   return failed.load() ? 1 : 0;
}

} // namespace

int main(int argc, char **argv) {
   try {
      if (argc == 1 || std::string(argv[1]) == "--help" ||
          std::string(argv[1]) == "help") {
         show_main_help(argv[0]);
         return 0;
      }
      if (std::string(argv[1]) == "--version") {
         std::cout << "SyclBenchmark " << tool_version << '\n';
         return 0;
      }
      const auto devices = discover_devices();
      if (devices.empty()) {
         throw std::runtime_error("no SYCL devices found");
      }
      const std::string command(argv[1]);
      if (command == "devices") {
         return run_devices_command(argc, argv, devices);
      }
      if (command == "benchmark") {
         return run_benchmark_command(argc, argv, devices);
      }
      if (command == "stress") {
         return run_stress_command(argc, argv, devices);
      }
      throw std::invalid_argument("unknown command: " + command);
   } catch (const std::exception &e) {
      std::cerr << "error: " << e.what() << '\n';
      return 2;
   }
}
