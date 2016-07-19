TARGETS = opencl_vector_add
CXXFLAGS = -Wall -std=c++1y -g

LDLIBS = -lOpenCL

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


all: $(TARGETS)


clean:
	$(RM) $(TARGETS)
