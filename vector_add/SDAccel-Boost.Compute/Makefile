# Adapt to the real platform
XOCCFLAGS=--platform xilinx:adm-pcie-7v3:1ddr:3.0

# To be sure to select a Xilinx platform, try for example this before
# running the application:
#export BOOST_COMPUTE_DEFAULT_VENDOR=Xilinx

TARGETS = vector_add

CXXFLAGS = -Wall -std=c++1y -g \
	-DBOOST_COMPUTE_DEBUG_KERNEL_COMPILATION \
	-DBOOST_COMPUTE_HAVE_THREAD_LOCAL \
	-DBOOST_COMPUTE_THREAD_SAFE

# -lpthread is required by Xilinx OpenCL implementation
LDLIBS = -lOpenCL -pthread

# Specify where OpenCL includes files are with OpenCL_INCPATH
ifdef OpenCL_INCPATH
	CXXFLAGS += -I$(OpenCL_INCPATH)
endif

# Specify where Boost.Compute is with BOOST_COMPUTE_INCPATH
ifdef BOOST_COMPUTE_INCPATH
	CXXFLAGS += -I$(BOOST_COMPUTE_INCPATH)
endif

# Specify where OpenCL library files are with OpenCL_LIBPATH
ifdef OpenCL_LIBPATH
  LDFLAGS += -L$(OpenCL_LIBPATH)
endif

%.xclbin: %.cl
	xocc $(XOCCFLAGS) --target hw -o $@ $<

# Build the host part and the kernel part
all: $(TARGETS) $(TARGETS).xclbin

clean:
	$(RM) $(TARGETS) $(TARGETS).xclbin
