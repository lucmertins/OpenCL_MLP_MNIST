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

#include "mlp.h"
#include "mlpTraining.h"

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "apiopencl.h"

using namespace std;

ifstream image;
ifstream label;
ofstream report;

cl_mem clIn2;

cl_context context = 0;
cl_command_queue commandQueue = 0;
cl_program program = 0;
cl_kernel kernelClearN2 = 0, kernelIncrementN2 = 0;
cl_mem memObjects[3] = {0, 0, 0};

void showDate()
{
    char result[100];
    time_t t = time(NULL);
    strftime(result, sizeof(result), "%d/%m/%Y %H:%M", localtime(&t));
    report << "Tempo:  " << result << endl;
}

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
        }
    }
}

void initOpenCL(int plataformId, cl_device_id *device)
{
    context = createContext(plataformId);
    avalError(context, 1, CL_SUCCESS);
    commandQueue = createCommandQueue(context, device);
    avalError(commandQueue, 2, CL_SUCCESS);
    program = createProgram(context, *device, "mlp.cl");
    avalError(program, 3, CL_SUCCESS);
    kernelClearN2 = clCreateKernel(program, "clearN2", NULL);
    avalError(kernelClearN2, 4, CL_SUCCESS);
    kernelIncrementN2 = clCreateKernel(program, "incrementN2", NULL);
    avalError(kernelIncrementN2, 44, CL_SUCCESS);
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

    clIn2 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double) *tam, in2, &errNum);

    size_t globalWorkSize[1] = {tam};
    size_t localWorkSize[1] = {1};
    avalError(context, 5, clSetKernelArg(kernelClearN2, 0, sizeof(cl_mem), &clIn2));
    avalError(context, 6, clEnqueueNDRangeKernel(commandQueue, kernelClearN2, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL));

    avalError(context, 55, clSetKernelArg(kernelIncrementN2, 0, sizeof(cl_mem), &clIn2));
    avalError(context, 66, clEnqueueNDRangeKernel(commandQueue, kernelIncrementN2, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL));

    // Read the output buffer back to the Host
    avalError(context, 7, clEnqueueReadBuffer(commandQueue, clIn2, CL_TRUE, 0, tam * sizeof(double), in2, 0, NULL, NULL));

    cout << "Result" << endl;
    for (int i = 0; i < tam; i++)
    {
        cout << in2[i] << " ";
    }
    cout << endl
         << "Executed program succesfully." << endl;

    // aboutTraining();
    // report.open(report_fn.c_str(), ios::out);
    // image.open(training_image_fn.c_str(), ios::in | ios::binary); // Binary image file
    // label.open(training_label_fn.c_str(), ios::in | ios::binary); // Binary label file

    // // Retirar headers do arquivo
    // char number;
    // for (int i = 1; i <= 16; ++i)
    // {
    //     image.read(&number, sizeof(char));
    // }
    // for (int i = 1; i <= 8; ++i)
    // {
    //     label.read(&number, sizeof(char));
    // }
    // initLayersRoundWeight();

    // showDate();

    // training();
    // saveMLP(model_fn);

    // showDate();
    // report.close();
    // image.close();
    // label.close();
    clReleaseMemObject(clIn2);
    clReleaseKernel(kernelIncrementN2);
    clReleaseKernel(kernelClearN2);
    clReleaseProgram(program);
    clReleaseCommandQueue(commandQueue);
    clReleaseContext(context);
}
