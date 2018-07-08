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

void showPlataforms()
{
    cl_int errNum;
    cl_uint numPlatforms;
    cl_platform_id allPlatformId[20];
    cl_context context = NULL;

    errNum = clGetPlatformIDs(20, allPlatformId, &numPlatforms);
    if (errNum != CL_SUCCESS || numPlatforms <= 0)
    {
        std::cerr << "Failed to find any OpenCL platforms." << std::endl;
        return;
    }
    else
    {
        std::cout << "PlataformId [" << numPlatforms << "]" << std::endl;
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
}
void showDevices(int plataformSelect)
{
    cl_int errNum;
    cl_uint numPlatforms;
    cl_platform_id allPlatformId[20];
    cl_context context = NULL;

    errNum = clGetPlatformIDs(20, allPlatformId, &numPlatforms);
    if (errNum != CL_SUCCESS || numPlatforms <= 0)
    {
        std::cerr << "Failed to find any OpenCL platforms." << std::endl;
        return;
    }
    cl_device_id devices[20];
    cl_uint devices_n = 0;
    errNum = clGetDeviceIDs(allPlatformId[plataformSelect], CL_DEVICE_TYPE_GPU, 20, devices, &devices_n);
    if (errNum != CL_SUCCESS || devices_n <= 0)
    {
        std::cout << "Err [" << errNum << "] Failed to find any OpenCL device" << std::endl;
        return;
    }
    else
    {
        std::cout << "Devices Plataform [" << plataformSelect << "]" << std::endl;
        for (int i = 0; i < devices_n; i++)
        {
            char buffer[10240];
            cl_uint buf_uint;
            cl_ulong buf_ulong;
            std::cout << "Devices [" << i << "]" << std::endl;
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
}

///
//  Create an OpenCL context on the plataformSelect
//  either a GPU or CPU depending on what is available.
//
cl_context createContext(int plataformSelect)
{
    cl_int errNum;
    cl_uint numPlatforms;
    cl_platform_id allPlatformId[20];
    cl_context context = NULL;

    errNum = clGetPlatformIDs(20, allPlatformId, &numPlatforms);
    if (errNum != CL_SUCCESS || numPlatforms <= 0)
    {
        std::cerr << "Failed to find any OpenCL platforms." << std::endl;
        return NULL;
    }
    cl_device_id devices[20];
    cl_uint devices_n = 0;
    errNum = clGetDeviceIDs(allPlatformId[plataformSelect], CL_DEVICE_TYPE_GPU, 20, devices, &devices_n);
    if (errNum != CL_SUCCESS || devices_n <= 0)
    {
        std::cout << "Err [" << errNum << "] Failed to find any OpenCL device on PlataformID [" << plataformSelect << "]" << std::endl;
        return NULL;
    }
    cl_context_properties contextProperties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)allPlatformId[plataformSelect], 0};
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
cl_command_queue createCommandQueue(cl_context context, cl_device_id *device)
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
        std::cerr << "Failed to get device IDs " << std::endl;
        return NULL;
    }

    // In this example, we just choose the first available device.  In a
    // real program, you would likely use all available devices or choose
    // the highest performance device based on OpenCL device queries
    commandQueue = clCreateCommandQueueWithProperties(context, devices[0], 0, &errNum);
    if (commandQueue == NULL)
    {
        delete[] devices;
        std::cerr << "Failed to create commandQueue for device 0 [" << errNum << "]" << std::endl;
        return NULL;
    }

    *device = devices[0];
    delete[] devices;
    return commandQueue;
}

///
//  Create an OpenCL program from the kernel source file
//
cl_program createProgram(cl_context context, cl_device_id device, const char *fileName)
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
    program = clCreateProgramWithSource(context, 1, (const char **)&srcStr, NULL, NULL);
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
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buildLog), buildLog, NULL);

        std::cerr << "Error in kernel: " << std::endl;
        std::cerr << buildLog;
        clReleaseProgram(program);
        return NULL;
    }

    return program;
}