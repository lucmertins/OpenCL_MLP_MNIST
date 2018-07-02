// mlp.cpp
//
//    Backpropagation MLP MNIST OpenCL

#include <iostream>
#include <fstream>
#include <sstream>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "apiopencl.h"

#define CL_CHECK(_expr)                                                             \
    do                                                                              \
    {                                                                               \
        cl_int _err = _expr;                                                        \
        if (_err == CL_SUCCESS)                                                     \
            break;                                                                  \
        std::cout << "OpenCL Error: " << #_expr << "returned " << (int)_err << "!"; \
        abort();                                                                    \
    } while (0)

///
//  Create an OpenCL context on the first available platform using
//  either a GPU or CPU depending on what is available.
//
cl_context CreateContext()
{
    cl_int errNum;
    cl_uint numPlatforms;
    cl_platform_id allPlatformId[20];
    cl_context context = NULL;

    // First, select an OpenCL platform to run on.  For this example, we
    // simply choose the first available platform.  Normally, you would
    // query for all available platforms and select the most appropriate one.
    errNum = clGetPlatformIDs(20, allPlatformId, &numPlatforms);
    if (errNum != CL_SUCCESS || numPlatforms <= 0)
    {
        std::cerr << "Failed to find any OpenCL platforms." << std::endl;
        return NULL;
    }
    else
    {
        std::cout << "Num Plataform [" << numPlatforms << "]" << std::endl;
        for (int i = 0; i < numPlatforms; i++)
        {
            char buffer[10240];
            std::cout << i << std::endl;
            clGetPlatformInfo(allPlatformId[i], CL_PLATFORM_PROFILE, 10240, buffer, NULL);
            std::cout << "  PROFILE =" << buffer << std::endl;
            clGetPlatformInfo(allPlatformId[i], CL_PLATFORM_VERSION, 10240, buffer, NULL);
            std::cout << "  VERSION =" << buffer << std::endl;
            clGetPlatformInfo(allPlatformId[i], CL_PLATFORM_NAME, 10240, buffer, NULL);
            std::cout << "  NAME =" << buffer << std::endl;
            clGetPlatformInfo(allPlatformId[i], CL_PLATFORM_VENDOR, 10240, buffer, NULL);
            std::cout << "  VENDOR =" << buffer << std::endl;
            clGetPlatformInfo(allPlatformId[i], CL_PLATFORM_EXTENSIONS, 10240, buffer, NULL);
            std::cout << "  EXTENSIONS =" << buffer << std::endl;
        }
    }

    cl_device_id devices[20];
    cl_uint devices_n = 0;
    errNum = clGetDeviceIDs(allPlatformId[2], CL_DEVICE_TYPE_GPU, 20, devices, &devices_n);
    if (errNum != CL_SUCCESS || devices_n <= 0)
    {
        std::cout << "Err [" << errNum << "] Failed to find any OpenCL device" << std::endl;
        return NULL;
    }
    else
    {
        for (int i = 0; i < devices_n; i++)
        {
            char buffer[10240];
            cl_uint buf_uint;
            cl_ulong buf_ulong;
            std::cout << i << std::endl;
            clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(buffer), buffer, NULL);
            std::cout << "  DEVICE_NAME =" << buffer << std::endl;
            clGetDeviceInfo(devices[i], CL_DEVICE_VENDOR, sizeof(buffer), buffer, NULL);
            std::cout << "  DEVICE_VENDOR =" << buffer << std::endl;
            clGetDeviceInfo(devices[i], CL_DEVICE_VERSION, sizeof(buffer), buffer, NULL);
            std::cout << "  DEVICE_VERSION =" << buffer << std::endl;
            clGetDeviceInfo(devices[i], CL_DRIVER_VERSION, sizeof(buffer), buffer, NULL);
            std::cout << "  DRIVER_VERSION =" << buffer << std::endl;
            clGetDeviceInfo(devices[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(buf_uint), &buf_uint, NULL);
            std::cout << "  DEVICE_MAX_COMPUTE_UNITS =" << buf_uint << std::endl;
            clGetDeviceInfo(devices[i], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(buf_uint), &buf_uint, NULL);
            std::cout << "  DEVICE_MAX_CLOCK_FREQUENCY =" << buf_uint << std::endl;
            clGetDeviceInfo(devices[i], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(buf_ulong), &buf_ulong, NULL);
            std::cout << "  DEVICE_GLOBAL_MEM_SIZE =" << buf_ulong << std::endl;
        }
    }
    std::cout << "vamo" << std::endl;
    // Next, create an OpenCL context on the platform.  Attempt to
    // create a GPU-based context, and if that fails, try to create
    // a CPU-based context.
    cl_context_properties contextProperties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)allPlatformId[2], 0};
    context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU, NULL, NULL, &errNum);
    if (errNum != CL_SUCCESS)
    {
        std::cout << "Err [" << errNum << "] Could not create GPU context, trying CPU..." << std::endl;
        context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_CPU, NULL, NULL, &errNum);
        if (errNum != CL_SUCCESS)
        {
            std::cerr << "Err [" << errNum << "] Failed to create an OpenCL GPU or CPU context." << std::endl;
            return NULL;
        }
    }
    return context;
}

///
//  Create a command queue on the first device available on the
//  context
//
cl_command_queue CreateCommandQueue(cl_context context, cl_device_id *device)
{
    cl_int errNum;
    cl_device_id *devices;
    cl_command_queue commandQueue = NULL;
    size_t deviceBufferSize = -1;

    // First get the size of the devices buffer
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Failed call to clGetContextInfo(...,GL_CONTEXT_DEVICES,...)";
        return NULL;
    }

    if (deviceBufferSize <= 0)
    {
        std::cerr << "No devices available.";
        return NULL;
    }

    // Allocate memory for the devices buffer
    devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);
    if (errNum != CL_SUCCESS)
    {
        delete[] devices;
        std::cerr << "Failed to get device IDs";
        return NULL;
    }

    // In this example, we just choose the first available device.  In a
    // real program, you would likely use all available devices or choose
    // the highest performance device based on OpenCL device queries
    commandQueue = clCreateCommandQueueWithProperties(context, devices[0], 0, NULL);
    if (commandQueue == NULL)
    {
        delete[] devices;
        std::cerr << "Failed to create commandQueue for device 0";
        return NULL;
    }

    *device = devices[0];
    delete[] devices;
    return commandQueue;
}

///
//  Create an OpenCL program from the kernel source file
//
cl_program CreateProgram(cl_context context, cl_device_id device, const char *fileName)
{
    cl_int errNum;
    cl_program program;

    std::ifstream kernelFile(fileName, std::ios::in);
    if (!kernelFile.is_open())
    {
        std::cerr << "Failed to open file for reading: " << fileName << std::endl;
        return NULL;
    }

    std::ostringstream oss;
    oss << kernelFile.rdbuf();

    std::string srcStdStr = oss.str();
    const char *srcStr = srcStdStr.c_str();
    program = clCreateProgramWithSource(context, 1,
                                        (const char **)&srcStr,
                                        NULL, NULL);
    if (program == NULL)
    {
        std::cerr << "Failed to create CL program from source." << std::endl;
        return NULL;
    }

    errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (errNum != CL_SUCCESS)
    {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                              sizeof(buildLog), buildLog, NULL);

        std::cerr << "Error in kernel: " << std::endl;
        std::cerr << buildLog;
        clReleaseProgram(program);
        return NULL;
    }

    return program;
}

///
//  Create memory objects used as the arguments to the kernel
//  The kernel takes three arguments: result (output), a (input),
//  and b (input)
//
bool CreateMemObjects(cl_context context, cl_mem memObjects[3], float *a, float *b)
{
    memObjects[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(float) * ARRAY_SIZE, a, NULL);
    memObjects[1] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(float) * ARRAY_SIZE, b, NULL);
    memObjects[2] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   sizeof(float) * ARRAY_SIZE, NULL, NULL);

    if (memObjects[0] == NULL || memObjects[1] == NULL || memObjects[2] == NULL)
    {
        std::cerr << "Error creating memory objects." << std::endl;
        return false;
    }

    return true;
}

///
//  Cleanup any created OpenCL resources
//
void Cleanup(cl_context context, cl_command_queue commandQueue, cl_program program, cl_kernel kernel, cl_mem memObjects[3])
{
    for (int i = 0; i < 3; i++)
    {
        if (memObjects[i] != 0)
            clReleaseMemObject(memObjects[i]);
    }
    if (commandQueue != 0)
        clReleaseCommandQueue(commandQueue);

    if (kernel != 0)
        clReleaseKernel(kernel);

    if (program != 0)
        clReleaseProgram(program);

    if (context != 0)
        clReleaseContext(context);
}
