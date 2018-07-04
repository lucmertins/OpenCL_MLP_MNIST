// iris.cpp
//
//    This is a simple example that demonstrates basic OpenCL setup and
//    use.

#include <iostream>
#include <fstream>
#include <sstream>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "apiopencl.h"

const int ARRAY_SIZE = 1000;
///
//  Create memory objects used as the arguments to the kernel
//  The kernel takes three arguments: result (output), a (input),
//  and b (input)
//
bool createMemObjects(cl_context context, cl_mem memObjects[3], float *a, float *b)
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
//	main() for HelloWorld example
//
int main(int argc, char **argv)
{
    cl_context context = 0;
    cl_command_queue commandQueue = 0;
    cl_program program = 0;
    cl_device_id device = 0;
    cl_kernel kernel = 0;
    cl_mem memObjects[3] = {0, 0, 0};
    cl_int errNum;
    
    int plataformId=2;
    showPlataforms();
    showDevices(plataformId);
    // As the result of the above function
    // 0 pc casa
    // 2 notebook com bumblebee
    std::cout << "****************" << std::endl;

    context = createContext(plataformId);
    if (context == NULL)
    {
        std::cerr << "Failed to create OpenCL context." << std::endl;
        return 1;
    }

    // Create a command-queue on the first device available
    // on the created context
    commandQueue = createCommandQueue(context, &device);
    if (commandQueue == NULL)
    {
        cleanup(context, commandQueue, program, kernel, memObjects);
        return 1;
    }

    // Create OpenCL program from HelloWorld.cl kernel source
    program = createProgram(context, device, "testeOkOpenCL.cl");
    if (program == NULL)
    {
        cleanup(context, commandQueue, program, kernel, memObjects);
        return 1;
    }

    // Create OpenCL kernel
    kernel = clCreateKernel(program, "hello_kernel", NULL);
    if (kernel == NULL)
    {
        std::cerr << "Failed to create kernel" << std::endl;
        cleanup(context, commandQueue, program, kernel, memObjects);
        return 1;
    }

    // Create memory objects that will be used as arguments to
    // kernel.  First create host memory arrays that will be
    // used to store the arguments to the kernel
    float result[ARRAY_SIZE];
    float a[ARRAY_SIZE];
    float b[ARRAY_SIZE];
    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        a[i] = (float)i;
        b[i] = (float)(i * 2);
    }

    if (!createMemObjects(context, memObjects, a, b))
    {
        cleanup(context, commandQueue, program, kernel, memObjects);
        return 1;
    }

    // Set the kernel arguments (result, a, b)
    errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &memObjects[0]);
    errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &memObjects[1]);
    errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &memObjects[2]);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error setting kernel arguments." << std::endl;
        cleanup(context, commandQueue, program, kernel, memObjects);
        return 1;
    }

    size_t globalWorkSize[1] = {ARRAY_SIZE};
    size_t localWorkSize[1] = {1};

    // Queue the kernel up for execution across the array
    errNum = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,
                                    globalWorkSize, localWorkSize,
                                    0, NULL, NULL);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error queuing kernel for execution." << std::endl;
        cleanup(context, commandQueue, program, kernel, memObjects);
        return 1;
    }

    // Read the output buffer back to the Host
    errNum = clEnqueueReadBuffer(commandQueue, memObjects[2], CL_TRUE,
                                 0, ARRAY_SIZE * sizeof(float), result,
                                 0, NULL, NULL);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error reading result buffer." << std::endl;
        cleanup(context, commandQueue, program, kernel, memObjects);
        return 1;
    }

    // Output the result buffer
    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Executed program succesfully." << std::endl;
    cleanup(context, commandQueue, program, kernel, memObjects);

    return 0;
}
