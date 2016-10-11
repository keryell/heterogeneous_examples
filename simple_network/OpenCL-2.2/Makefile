TARGETS = simple_network
CXXFLAGS = -Wall -std=c++1y -g -I../../include -I. \
	-DBOOST_COMPUTE_DEBUG_KERNEL_COMPILATION \
	-DBOOST_COMPUTE_HAVE_THREAD_LOCAL \
	-DBOOST_COMPUTE_THREAD_SAFE

LDLIBS = -lOpenCL

# Specify where OpenCL includes files are with OpenCL_INCPATH
ifdef OpenCL_INCPATH
	CXXFLAGS += -I$(OpenCL_INCPATH)
endif

# Specify where Bost.Compute is with BOOST_COMPUTE_INCPATH
ifdef BOOST_COMPUTE_INCPATH
	CXXFLAGS += -I$(BOOST_COMPUTE_INCPATH)
endif

# Specify where OpenCL library files are with OpenCL_LIBPATH
ifdef OpenCL_LIBPATH
  LDFLAGS += -L$(OpenCL_LIBPATH)
endif


all: $(TARGETS)
# simple_network.spv simple_network.spv-text


clean:
	$(RM) $(TARGETS)

# Keep the intermediate bitcode for inspection
.PRECIOUS: %.bc

%.bc: %.cl++
	clang -cc1 -triple spir64-unknown-unknown -cl-std=CL2.1 -x cl -emit-llvm -I ~/AMD/Projects/OpenCL/Intel_C++_OpenSource/libclcxx/include $<
	llvm-as < $*.ll > $*.bc

%.spv: %.bc
	llvm-spirv -o $@ $*.bc

%.spv-text: %.bc
	llvm-spirv -spirv-text -o $@ $*.bc


dist:
	tar zcvf ../samples.tgz Makefile *.{bc,cl,cl++,cpp,ll,spv,spv-text} README.rst
