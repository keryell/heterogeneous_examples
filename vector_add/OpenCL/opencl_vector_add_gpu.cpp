#include <iostream>
#include <string>
#include <vector>

#if defined(__APPLE__)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

/* Transform the value of a given symbol to a string. Since we expect a
   macro symbol, use a double evaluation... */
#define _strinG(s) #s

#define _stringifY(s) _strinG(s)

/** Throw a nicer error message in the code by adding the file name and
    the position */
#define THROW_ERROR(message)                                            \
  throw std::domain_error(std::string("In file " __FILE__ " at line "   \
                                      _stringifY(__LINE__) "\n") + message)

/** Test for an OpenCL error and display a message */
#define OCL_TEST_ERROR_MSG(status, msg) do {                            \
    if ((status) !=  CL_SUCCESS)                                        \
      THROW_ERROR(std::string(msg) + std::to_string(status));           \
  } while(0)

  /** Do an OpenCL function call and test for execution error */
#define OCL_ERROR(func) do {                                            \
    cl_int _st = func;                                                  \
    if (_st !=  CL_SUCCESS)                                             \
      THROW_ERROR(_stringifY(func) " returns error " + std::to_string(_st)); \
  } while(0)

constexpr size_t N = 3;

using Vector = float[N];

int main() {
  Vector a = { 1, 2, 3 };
  Vector b = { 5, 6, 8 };
  Vector c;

  cl_int status;

  // Get the number of OpenCL platforms on the machine
  cl_uint num_platforms;
  OCL_ERROR(clGetPlatformIDs(0, NULL, &num_platforms));

  std::vector<cl_platform_id> platforms(num_platforms);
  OCL_ERROR(clGetPlatformIDs(num_platforms, platforms.data(), NULL));


  cl_context context;
  bool found_context = false;
  for (auto platform : platforms) {
    std::cout << platform << std::endl;
    // Describe the context to query
    cl_context_properties cps[] = {
      CL_CONTEXT_PLATFORM, (cl_context_properties)platform,
      0
    };
    // Create an OpenCL context from our platform
    context = clCreateContextFromType(cps,
                                      CL_DEVICE_TYPE_GPU,
                                      NULL,
                                      NULL,
                                      &status);
    if (status ==  CL_SUCCESS) {
      found_context = true;
      break;
    }
  }
  if (!found_context)
    THROW_ERROR("Cannot found a context");

  // Get the first device
  cl_device_id device;
  OCL_ERROR(clGetContextInfo(context, CL_CONTEXT_DEVICES,
                             sizeof(device), &device, NULL));

  // Create an OpenCL command queue
  cl_command_queue command_queue =
    clCreateCommandQueueWithProperties(context, device, NULL, &status);
  OCL_TEST_ERROR_MSG(status, "Cannot create the command queue");

  // The input buffers for OpenCL
   cl_mem buffer_a =
     clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(a), NULL, &status);
   OCL_TEST_ERROR_MSG(status, "Cannot create buffer_a");
   cl_mem buffer_b =
     clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(b), NULL, &status);
   OCL_TEST_ERROR_MSG(status, "Cannot create buffer_b");

  // The output buffer for OpenCL
   cl_mem buffer_c =
     clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(c), NULL, &status);
   OCL_TEST_ERROR_MSG(status, "Cannot create buffer_c");

  // Construct an OpenCL program from the source file
  const char kernel_source[] = R"(
__kernel void vector_add(const __global float *a,
                         const __global float *b,
                         __global float *c) {
  c[get_global_id(0)] = a[get_global_id(0)] + b[get_global_id(0)];
}
)";
  const char *kernel_sources = kernel_source;
  const size_t kernel_size = sizeof(kernel_source);
  cl_program program = clCreateProgramWithSource(context, 1, &kernel_sources,
                                                 &kernel_size, &status);
  OCL_TEST_ERROR_MSG(status, "Cannot create program");

  OCL_ERROR(clBuildProgram(program, 1, &device, "", NULL, NULL));

  cl_kernel kernel = clCreateKernel(program, "vector_add", &status);
  OCL_TEST_ERROR_MSG(status, "Cannot find the kernel");

  // Send the input data to the accelerator
  OCL_ERROR(clEnqueueWriteBuffer(command_queue, buffer_a, true, 0 /* Offset */,
                                 sizeof(a), &a[0], 0, NULL, NULL));
  OCL_ERROR(clEnqueueWriteBuffer(command_queue, buffer_b, true, 0 /* Offset */,
                                 sizeof(b), &b[0], 0, NULL, NULL));

  OCL_ERROR(clSetKernelArg(kernel, 0, sizeof(buffer_a), &buffer_a));
  OCL_ERROR(clSetKernelArg(kernel, 1, sizeof(buffer_b), &buffer_b));
  OCL_ERROR(clSetKernelArg(kernel, 2, sizeof(buffer_c), &buffer_c));

  // Launch the kernel
  const size_t global_work_size { N };
  OCL_ERROR(clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
                                   &global_work_size, NULL,
                                   0, NULL, NULL));

  // Get the output data from the accelerator
  OCL_ERROR(clEnqueueReadBuffer(command_queue, buffer_c, true, 0 /* Offset */,
                                sizeof(c), &c[0], 0, NULL, NULL));


  std::cout << std::endl << "Result:" << std::endl;
  for(auto e : c)
    std::cout << e << " ";
  std::cout << std::endl;
}
































