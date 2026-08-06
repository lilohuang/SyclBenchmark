BACKEND  ?= auto
SYCL_ROOT ?= $(HOME)/sycl_workspace/llvm/build
CXX       := $(SYCL_ROOT)/bin/clang++

VALID_BACKENDS := auto cuda hip spirv all
ifeq ($(filter $(BACKEND),$(VALID_BACKENDS)),)
$(error BACKEND must be one of: $(VALID_BACKENDS))
endif

NVIDIA_TARGET := nvptx64-nvidia-cuda
SPIRV_TARGET  := spir64
AMD_TARGET    := amdgcn-amd-amdhsa

AMD_AGENT_ENUMERATOR ?= rocm_agent_enumerator
AMD_DETECTED_ARCH    := $(firstword $(filter-out gfx000, \
	$(filter gfx%, $(shell $(AMD_AGENT_ENUMERATOR) 2>/dev/null))))
AMD_GPU_ARCH         ?= $(AMD_DETECTED_ARCH)
CUDA_COMPUTE_CAP     := $(firstword $(shell nvidia-smi \
	--query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d '.'))
CUDA_DETECTED_ARCH   := $(if $(CUDA_COMPUTE_CAP),sm_$(CUDA_COMPUTE_CAP))
CUDA_ARCH            ?= $(CUDA_DETECTED_ARCH)
CUDA_SM              := $(shell printf '%s' '$(CUDA_ARCH)' | \
	sed -n 's/^sm_\([0-9][0-9]*\).*$$/\1/p')

ENABLE_NVIDIA := 0
ENABLE_AMD    := 0
ENABLE_SPIRV  := 0

ifeq ($(BACKEND),auto)
ENABLE_NVIDIA := $(if $(CUDA_DETECTED_ARCH),1,0)
ENABLE_AMD    := $(if $(AMD_DETECTED_ARCH),1,0)
ENABLE_SPIRV  := 1
endif
ifeq ($(BACKEND),cuda)
ENABLE_NVIDIA := 1
endif
ifeq ($(BACKEND),hip)
ENABLE_AMD := 1
endif
ifeq ($(BACKEND),spirv)
ENABLE_SPIRV := 1
endif
ifeq ($(BACKEND),all)
ENABLE_NVIDIA := 1
ENABLE_AMD    := 1
ENABLE_SPIRV  := 1
endif

empty :=
space := $(empty) $(empty)
comma := ,

ENABLED_TARGETS :=
ifeq ($(ENABLE_NVIDIA),1)
ENABLED_TARGETS += $(NVIDIA_TARGET)
endif
ifeq ($(ENABLE_SPIRV),1)
ENABLED_TARGETS += $(SPIRV_TARGET)
endif
ifeq ($(ENABLE_AMD),1)
ENABLED_TARGETS += $(AMD_TARGET)
endif
ENABLED_TARGETS := $(strip $(ENABLED_TARGETS))
OFFLOAD_TARGETS := $(subst $(space),$(comma),$(ENABLED_TARGETS))

ifeq ($(strip $(OFFLOAD_TARGETS)),)
$(error BACKEND=$(BACKEND) did not select any SYCL target)
endif

ifeq ($(ENABLE_NVIDIA),1)
ifeq ($(strip $(CUDA_ARCH)),)
$(error No NVIDIA GPU architecture detected; set CUDA_ARCH, for example sm_86)
endif
ifeq ($(strip $(CUDA_SM)),)
$(error Invalid CUDA_ARCH=$(CUDA_ARCH); expected a value such as sm_86)
endif
endif
ifeq ($(ENABLE_AMD),1)
ifeq ($(strip $(AMD_GPU_ARCH)),)
$(error No AMD GPU architecture detected; set AMD_GPU_ARCH, for example gfx1102)
endif
endif

REAL_RESOURCE_DIR := $(shell $(CXX) -print-resource-dir 2>/dev/null)
AMD_LIBSPIRV_NAME := libspirv.l64.signed_char.bc
AMD_LIBSPIRV      ?= $(REAL_RESOURCE_DIR)/lib/$(AMD_TARGET)-llvm/$(AMD_LIBSPIRV_NAME)
AMD_LIBSPIRV_DIR  := $(patsubst %/,%,$(abspath $(dir $(AMD_LIBSPIRV))))
CXX_ID             := $(shell $(CXX) --version 2>/dev/null | sha256sum | cut -c1-16)
AMD_LIBSPIRV_INPUT := $(wildcard $(AMD_LIBSPIRV))
CONFIG_KEY        := $(shell printf '%s' \
	'$(BACKEND)|$(CXX)|$(CXX_ID)|$(OFFLOAD_TARGETS)|$(CUDA_ARCH)|$(AMD_GPU_ARCH)|$(AMD_LIBSPIRV)' \
	| sha256sum | cut -c1-16)
RESOURCE_OVERLAY  := $(CURDIR)/build/clang-resource-$(CONFIG_KEY)

CXXFLAGS := -std=c++17 -O3 -fsycl -fsycl-targets=$(OFFLOAD_TARGETS)
LDFLAGS  := -Wl,-rpath,$(SYCL_ROOT)/lib

ifeq ($(ENABLE_NVIDIA),1)
CXXFLAGS += -Xsycl-target-backend=$(NVIDIA_TARGET) \
	--cuda-gpu-arch=$(CUDA_ARCH) \
	-DSYCL_BENCH_COMPILED_CUDA_SM=$(CUDA_SM)
endif
ifeq ($(ENABLE_SPIRV),1)
CXXFLAGS += -DSYCL_BENCH_COMPILED_SPIRV=1
endif
ifeq ($(ENABLE_AMD),1)
CXXFLAGS += -resource-dir=$(RESOURCE_OVERLAY) \
	-Xsycl-target-backend=$(AMD_TARGET) --offload-arch=$(AMD_GPU_ARCH)
ifneq ($(filter $(AMD_GPU_ARCH),gfx1100 gfx1101 gfx1102),)
CXXFLAGS += -DSYCL_BENCH_COMPILED_AMD_GFX11=1
endif
ifeq ($(AMD_GPU_ARCH),gfx90a)
CXXFLAGS += -DSYCL_BENCH_COMPILED_AMD_GFX90A=1
endif
LDFLAGS += -Xoffload-linker=$(AMD_TARGET) \
	'--lto-newpm-passes=globaloffset,lto<O3>'
endif

DEVICE_SELECTORS :=
ifeq ($(ENABLE_NVIDIA),1)
DEVICE_SELECTORS += cuda:*
endif
ifeq ($(ENABLE_SPIRV),1)
DEVICE_SELECTORS += level_zero:* opencl:*
endif
ifeq ($(ENABLE_AMD),1)
DEVICE_SELECTORS += hip:*
endif
ONEAPI_DEVICE_SELECTOR ?= $(subst $(space),;,$(strip $(DEVICE_SELECTORS)))

TARGET       := sycl-bench
BUILD_CONFIG := $(CURDIR)/build/.sycl-build-config-$(CONFIG_KEY)

ifneq ($(strip $(LD_LIBRARY_PATH)),)
export LD_LIBRARY_PATH := $(SYCL_ROOT)/lib:$(LD_LIBRARY_PATH)
else
export LD_LIBRARY_PATH := $(SYCL_ROOT)/lib
endif
export ONEAPI_DEVICE_SELECTOR

.DEFAULT_GOAL := $(TARGET)
.PHONY: run smoke clean config auto cuda hip spirv all

$(TARGET): main.cpp Makefile $(BUILD_CONFIG) $(if $(filter 1,$(ENABLE_AMD)),$(RESOURCE_OVERLAY)/.ready)
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS) -o $@

run: $(TARGET)
	./$(TARGET) devices

smoke: $(TARGET)
	python3 tests/smoke.py ./$(TARGET)

config:
	@echo "Build backend : $(BACKEND)"
	@echo "SYCL targets  : $(OFFLOAD_TARGETS)"
	@echo "Device filter : $(ONEAPI_DEVICE_SELECTOR)"
ifeq ($(ENABLE_NVIDIA),1)
	@echo "CUDA target   : $(CUDA_ARCH)"
endif
ifeq ($(ENABLE_AMD),1)
	@echo "AMD target    : $(AMD_GPU_ARCH)"
	@echo "AMD libspirv  : $(AMD_LIBSPIRV)"
endif

$(RESOURCE_OVERLAY)/.ready: Makefile $(BUILD_CONFIG) $(AMD_LIBSPIRV_INPUT) | $(CURDIR)/build
ifeq ($(ENABLE_AMD),1)
	@if [ ! -f "$(AMD_LIBSPIRV)" ]; then \
		echo "AMD libspirv not found: $(AMD_LIBSPIRV)"; \
		echo "Set AMD_LIBSPIRV to the signed-char bitcode file."; \
		exit 2; \
	fi
	@mkdir -p "$(RESOURCE_OVERLAY)/lib"
	@ln -sfn "$(REAL_RESOURCE_DIR)/include" "$(RESOURCE_OVERLAY)/include"
	@for source in "$(REAL_RESOURCE_DIR)"/lib/*; do \
		name="$${source##*/}"; \
		if [ "$$name" != "$(AMD_TARGET)" ]; then \
			ln -sfn "$$source" "$(RESOURCE_OVERLAY)/lib/$$name"; \
		fi; \
	done
	@if [ -L "$(RESOURCE_OVERLAY)/lib/$(AMD_TARGET)" ]; then \
		rm -f "$(RESOURCE_OVERLAY)/lib/$(AMD_TARGET)"; \
	fi
	@mkdir -p "$(RESOURCE_OVERLAY)/lib/$(AMD_TARGET)"
	@for source in "$(AMD_LIBSPIRV_DIR)"/*; do \
		name="$${source##*/}"; \
		if [ "$$name" != "$(AMD_LIBSPIRV_NAME)" ]; then \
			ln -sfn "$$source" \
				"$(RESOURCE_OVERLAY)/lib/$(AMD_TARGET)/$$name"; \
		fi; \
	done
	@$(CXX) -target $(AMD_TARGET) -nogpulib -Wno-override-module \
		-c -emit-llvm "$(AMD_LIBSPIRV)" \
		-o "$(RESOURCE_OVERLAY)/lib/$(AMD_TARGET)/$(AMD_LIBSPIRV_NAME)"
	@touch $@
endif

$(BUILD_CONFIG): | $(CURDIR)/build
	@printf '%s\n' \
		"BACKEND=$(BACKEND)" \
		"CXX=$(CXX)" \
		"CXXFLAGS=$(CXXFLAGS)" \
		"LDFLAGS=$(LDFLAGS)" > "$@"

$(CURDIR)/build:
	mkdir -p $@

auto cuda hip spirv all:
	$(MAKE) BACKEND=$@ $(TARGET)

clean:
	rm -f $(TARGET)
	rm -rf build
