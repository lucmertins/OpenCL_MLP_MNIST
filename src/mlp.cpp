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

// Output layer
double *in3, *out3, *theta3;

double expected[n3 + 1];

ifstream image;
ifstream label;
ofstream report;

cl_context context = 0;
cl_command_queue commandQueue = 0;
cl_program program = 0;
cl_kernel kernel = 0;
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

void initOpenCL(cl_device_id *device)
{
    //showPlataforms();
    // As the result of the above function
    // 0 pc casa
    // 2 notebook com bumblebee
    context = createContext(0);
    if (context == NULL)
    {
        std::cerr << "Failed to create OpenCL context." << std::endl;
        exit(-1);
    }
    commandQueue = createCommandQueue(context, device);
    if (commandQueue == NULL)
    {
        //   Cleanup(context, commandQueue, program, kernel, memObjects);
        exit(-1);
    }

    program = createProgram(context, *device, "testeOkOpenCL.cl");
    if (program == NULL)
    {
        cleanup(context, commandQueue, program, kernel, memObjects);
        exit(-1);
    }
    // Create OpenCL kernel
    kernel = clCreateKernel(program, "hello_kernel", NULL);
    if (kernel == NULL)
    {
        std::cerr << "Failed to create kernel" << std::endl;
        cleanup(context, commandQueue, program, kernel, memObjects);
        exit(-1);
    }
}

bool createMemObjects(cl_context context, int size, cl_mem memObjects[3], float *a, float *b)
{

    memObjects[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(float) * size, a, NULL);
    memObjects[1] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(float) * size, b, NULL);
    memObjects[2] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   sizeof(float) * size, NULL, NULL);

    if (memObjects[0] == NULL || memObjects[1] == NULL || memObjects[2] == NULL)
    {
        std::cerr << "Error creating memory objects." << std::endl;
        return false;
    }

    return true;
}

int main(int argc, char *argv[])
{

    cl_device_id device = 0;
    int ARRAY_SIZE = 1000;
    float result[ARRAY_SIZE];
    float a[ARRAY_SIZE];
    float b[ARRAY_SIZE];
    cl_int errNum;

    initOpenCL(&device);
    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        a[i] = (float)i;
        b[i] = (float)(i * 2);
    }
    if (!createMemObjects(context, ARRAY_SIZE, memObjects, a, b))
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
    size_t globalWorkSize[1] = {(size_t)ARRAY_SIZE};
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

    cleanup(context, commandQueue, program, kernel, memObjects);
}
