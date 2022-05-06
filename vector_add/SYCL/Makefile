# To use the DPC++ compiler:
SYCL_HOME=~/Xilinx/Projects/LLVM/worktrees/xilinx
export LD_LIBRARY_PATH=$SYCL_HOME/llvm/build/lib:$LD_LIBRARY_PATH

TARGETS = vector_add_XRT_interoperability

CXXFLAGS = -Wall -std=c++2b -g -g3 -fsycl -fsycl-targets=fpga64_hls_hw_emu \
  -I/opt/xilinx/xrt/include -L/opt/xilinx/xrt/lib -lOpenCL -luuid -lxrt_coreutil

LDLIBS =

# Specify where OpenCL includes files are with OpenCL_INCPATH
ifdef OpenCL_INCPATH
	CXXFLAGS += -I$(OpenCL_INCPATH)
endif

# Specify where CLHPP is with CLHPP_INCPATH
ifdef CLHPP_INCPATH
	CXXFLAGS += -I$(CLHPP_INCPATH)
endif

# Specify where OpenCL library files are with OpenCL_LIBPATH
ifdef OpenCL_LIBPATH
  LDFLAGS += -L$(OpenCL_LIBPATH)
endif

vector_add_XRT_interoperability: vector_add_XRT_interoperability.cpp
	$(SYCL_HOME)/llvm/build/bin/clang++ $(CXXFLAGS) -fsycl $< -o $@

all: $(TARGETS)


clean:
	$(RM) $(TARGETS)
