#include <iostream>
#include <fstream>
#include <cstring>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <set>
#include <iterator>
#include <algorithm>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "apiopencl.h"

using namespace std;

cl_mem clIn2;

cl_context context = 0;
cl_command_queue commandQueue = 0;
cl_program program = 0;
cl_kernel kernelClearN2 = 0, kernelIncrementN2 = 0, kernelMatriz2D = 0;
cl_mem memObjects[3] = {0, 0, 0};

void avalError(void *instance, int pos, cl_int errNum)
{
    if (instance == NULL || errNum != CL_SUCCESS)
    {
        switch (pos)
        {
        case 1:
            cout << "Failed to create OpenCL context." << endl;
            exit(pos);
        case 2:
            cout << "Failed to create Queue." << endl;
            clReleaseContext(context);
            exit(pos);
        case 3:
            cout << "Failed to create Program." << endl;
            clReleaseCommandQueue(commandQueue);
            clReleaseContext(context);
            exit(pos);
        case 4:
            cout << "Failed to create kernel" << endl;
            clReleaseProgram(program);
            clReleaseCommandQueue(commandQueue);
            clReleaseContext(context);
            exit(pos);
        case 44:
            cout << "Failed to create kernel" << endl;
            clReleaseKernel(kernelClearN2);
            clReleaseProgram(program);
            clReleaseCommandQueue(commandQueue);
            clReleaseContext(context);
            exit(pos);
        case 444:
            cout << "Failed to create kernel" << endl;
            clReleaseProgram(program);
            clReleaseCommandQueue(commandQueue);
            clReleaseContext(context);
            exit(pos);
        case 5:
            cout << "Error setting kernel arguments [" << errNum << "]" << endl;
            clReleaseMemObject(clIn2);
            clReleaseKernel(kernelIncrementN2);
            clReleaseKernel(kernelClearN2);
            clReleaseProgram(program);
            clReleaseCommandQueue(commandQueue);
            clReleaseContext(context);
            exit(pos);
        case 6:
            cout << "Error queuing kernel for execution. Errnum [" << errNum << "]" << endl;
            clReleaseMemObject(clIn2);
            clReleaseKernel(kernelIncrementN2);
            clReleaseKernel(kernelClearN2);
            clReleaseProgram(program);
            clReleaseCommandQueue(commandQueue);
            clReleaseContext(context);
            exit(pos);
        case 55:
            cout << "Error setting kernel arguments [" << errNum << "]" << endl;
            clReleaseMemObject(clIn2);
            clReleaseKernel(kernelIncrementN2);
            clReleaseKernel(kernelClearN2);
            clReleaseProgram(program);
            clReleaseCommandQueue(commandQueue);
            clReleaseContext(context);
            exit(pos);
        case 555:
            cout << "Error setting kernel arguments [" << errNum << "]" << endl;
            clReleaseKernel(kernelMatriz2D);
            clReleaseProgram(program);
            clReleaseCommandQueue(commandQueue);
            clReleaseContext(context);
            exit(pos);

        case 66:
            cout << "Error queuing kernel for execution. Errnum [" << errNum << "]" << endl;
            clReleaseMemObject(clIn2);
            clReleaseKernel(kernelIncrementN2);
            clReleaseKernel(kernelClearN2);
            clReleaseProgram(program);
            clReleaseCommandQueue(commandQueue);
            clReleaseContext(context);
            exit(pos);

        case 7:
            cout << "Error reading result buffer. Errnum [" << errNum << "]" << endl;
            clReleaseMemObject(clIn2);
            clReleaseKernel(kernelIncrementN2);
            clReleaseKernel(kernelClearN2);
            clReleaseProgram(program);
            clReleaseCommandQueue(commandQueue);
            clReleaseContext(context);
            exit(pos);
        default:
            cout << "Error unknown. Pos [" << pos << "]. Errnum [" << errNum << "]" << endl;
            exit(pos);
        }
    }
}

void initOpenCL(int plataformId, cl_device_id *device)
{
    context = createContext(plataformId);
    avalError(context, 1, CL_SUCCESS);
    commandQueue = createCommandQueue(context, device);
    avalError(commandQueue, 2, CL_SUCCESS);
    program = createProgram(context, *device, "testeMatrizes.cl");
    avalError(program, 3, CL_SUCCESS);
    kernelClearN2 = clCreateKernel(program, "clearN2", NULL);
    avalError(kernelClearN2, 4, CL_SUCCESS);
    kernelIncrementN2 = clCreateKernel(program, "incrementN2", NULL);
    avalError(kernelIncrementN2, 44, CL_SUCCESS);
    kernelMatriz2D = clCreateKernel(program, "matriz2D", NULL);
    avalError(kernelMatriz2D, 444, CL_SUCCESS);
}

int main(int argc, char *argv[])
{

    cl_device_id device = 0;
    cl_int errNum;

    //showPlataforms();
    // As the result of the above function
    // 0 pc casa
    // 2 notebook com bumblebee
    int plataformId = 0;

    initOpenCL(plataformId, &device);

    int tam = 60000;
    double *in2 = new double[tam];
    cout << "Buffer creating." << endl;

    clIn2 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double) * tam, in2, &errNum);

    size_t globalWorkSize[1] = {(size_t)tam};
    size_t localWorkSize[1] = {1};
    avalError(context, 5, clSetKernelArg(kernelClearN2, 0, sizeof(cl_mem), &clIn2));
    avalError(context, 6, clEnqueueNDRangeKernel(commandQueue, kernelClearN2, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL));

    avalError(context, 55, clSetKernelArg(kernelIncrementN2, 0, sizeof(cl_mem), &clIn2));
    avalError(context, 66, clEnqueueNDRangeKernel(commandQueue, kernelIncrementN2, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL));

    // Read the output buffer back to the Host
    avalError(context, 7, clEnqueueReadBuffer(commandQueue, clIn2, CL_TRUE, 0, tam * sizeof(double), in2, 0, NULL, NULL));

    // cout << "Result" << endl;
    // for (int i = 0; i < tam; i++)
    // {
    //     cout << in2[i] << " ";
    // }
    // cout << endl
    //      << endl
    //      << endl
    //      << endl;

    clReleaseMemObject(clIn2);
    clReleaseKernel(kernelIncrementN2);
    clReleaseKernel(kernelClearN2);

    int width = 10, height = 10;

    globalWorkSize[0] = (size_t)(width);
    localWorkSize[0] = (size_t)(height);
    cl_mem device_array = clCreateBuffer(context, CL_MEM_READ_WRITE, width * height * sizeof(int), NULL, NULL);
    avalError(context, 555, clSetKernelArg(kernelMatriz2D, 0, sizeof(cl_mem), &device_array));
    avalError(context, 555, clSetKernelArg(kernelMatriz2D, 1, sizeof(cl_int), &width));
    avalError(context, 555, clSetKernelArg(kernelMatriz2D, 2, sizeof(cl_int), &height));
    avalError(context, 666, clEnqueueNDRangeKernel(commandQueue, kernelMatriz2D, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL));

    cout << endl
         << "Mostrar matriz 2d" << endl;

    int *ptr_to_array_data = (int *)malloc(sizeof(int) * width * height);
    avalError(context, 777, clEnqueueReadBuffer(commandQueue, device_array, CL_TRUE, 0, width * height * sizeof(int), ptr_to_array_data, 0, NULL, NULL));
    for (int h = 0; h < height; h++)
    {
        for (int w = 0; w < width; w++)

        {
            cout << *(ptr_to_array_data + (w + h * width)) << " ";
        }
        cout << endl;
    }
    cout << endl;

    cout << *ptr_to_array_data << " ";
    cout << *(ptr_to_array_data + 5) << " ";

    cout << "Fim" << endl;
    free(ptr_to_array_data);
    clReleaseMemObject(device_array);
    clReleaseKernel(kernelMatriz2D);
    clReleaseProgram(program);
    clReleaseCommandQueue(commandQueue);
    clReleaseContext(context);
}
