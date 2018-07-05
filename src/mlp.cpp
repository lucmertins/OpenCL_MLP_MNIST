#include <iostream>
#include <fstream>
#include <cstring>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cmath>
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

// Image size in MNIST database
const int width = 28;
const int height = 28;

// n1 = Number of input neurons
// n2 = Number of hidden neurons
// n3 = Number of output neurons
// epochs = Number of iterations for back-propagation algorithm
// learning_rate = Learing rate
// momentum = Momentum (heuristics to optimize back-propagation algorithm)
// epsilon = Epsilon, no more iterations if the learning error is smaller than epsilon

const int n1 = width * height; // = 784
const int n2 = 128;            // hidden layer
const int n3 = 10;             // Ten classes: 0 - 9
const int epochs = 512;
const double learning_rate = 1e-3;
const double momentum = 0.9;
const double epsilon = 1e-3;

// MLP Definition
// input layer
double *w1[n1 + 1], *delta1[n1 + 1], *out1;

// hidden layer
double *w2[n2 + 1], *delta2[n2 + 1], *in2, *out2, *theta2;
cl_mem clIn2;

// Output layer
double *in3, *out3, *theta3;

double expected[n3 + 1];

ifstream image;
ifstream label;
ofstream report;

cl_context context = 0;
cl_command_queue commandQueue = 0;
cl_program program = 0;
cl_kernel kernelClearN2 = 0, kernelIncrementN2 = 0;
cl_mem memObjects[3] = {0, 0, 0};

double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

void processPerceptron()
{
    for (int i = 1; i <= n2; ++i)
    {
        in2[i] = 0.0;
    }
    for (int i = 1; i <= n3; ++i)
    {
        in3[i] = 0.0;
    }
    for (int i = 1; i <= n1; ++i)
    {
        for (int j = 1; j <= n2; ++j)
        {
            in2[j] += out1[i] * w1[i][j];
        }
    }
    for (int i = 1; i <= n2; ++i)
    {
        out2[i] = sigmoid(in2[i]);
    }
    for (int i = 1; i <= n2; ++i)
    {
        for (int j = 1; j <= n3; ++j)
        {
            in3[j] += out2[i] * w2[i][j];
        }
    }
    for (int i = 1; i <= n3; ++i)
    {
        out3[i] = sigmoid(in3[i]);
    }
}

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

    in2 = new double[n2 + 1];
    for (int i = 0; i < n2 + 1; i++)
    {
        cout << in2[i] << " ";
    }
    cout << endl
         << endl
         << "Buffer creating." << endl;

    clIn2 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double) * (n2 + 1), in2, &errNum);

    size_t globalWorkSize[1] = {n2 + 1};
    size_t localWorkSize[1] = {1};
    avalError(context, 5, clSetKernelArg(kernelClearN2, 0, sizeof(cl_mem), &clIn2));
    avalError(context, 6, clEnqueueNDRangeKernel(commandQueue, kernelClearN2, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL));

    avalError(context, 55, clSetKernelArg(kernelIncrementN2, 0, sizeof(cl_mem), &clIn2));
    avalError(context, 66, clEnqueueNDRangeKernel(commandQueue, kernelIncrementN2, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL));

    // Read the output buffer back to the Host
    avalError(context, 7, clEnqueueReadBuffer(commandQueue, clIn2, CL_TRUE, 0, (n2 + 1) * sizeof(double), in2, 0, NULL, NULL));

    cout << endl
         << endl;
    // Output the result buffer
    for (int i = 0; i < n2 + 1; i++)
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

    cleanup(context, commandQueue, program, kernelClearN2, memObjects);
}
