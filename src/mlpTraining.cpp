#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "mlpTraining.h"
#include "apiopencl.h"

using namespace std;

// Image size in MNIST database
const int width = 28;
const int height = 28;

const int n1 = width * height; // = 784
const int n2 = 128;            // hidden layer
const int n3 = 10;             // Ten classes: 0 - 9
const int epochs = 512;
const double learning_rate = 1e-3;
const double momentum = 0.9;
const double epsilon = 1e-3;

// MLP Definition
// input layer
double *pW1 = 0, *pDelta1 = 0, *pOut1 = 0;

// hidden layer
double *pW2 = 0, *pDelta2 = 0, *pIn2 = 0, *pOut2 = 0, *pTheta2 = 0;

// Output layer
double *pIn3 = 0, *pOut3 = 0, *pTheta3 = 0;

double *pExpected = 0;

// Numero de exemplos
const int nTraining = 60000;

cl_context context = 0;
cl_command_queue commandQueue = 0;
cl_program program = 0;
cl_kernel kernelClearVector = 0;
cl_kernel kernelSumVector = 0;
cl_kernel kernelSigmoid = 0;

cl_device_id *deviceGlobal;

void aboutTraining()
{
    cout << "**************************************************" << endl;
    cout << "*** Training Neural Network for MNIST database ***" << endl;
    cout << "**************************************************" << endl;
    cout << endl;
    cout << "No. input neurons: " << n1 << endl;
    cout << "No. hidden neurons: " << n2 << endl;
    cout << "No. output neurons: " << n3 << endl;
    cout << endl;
    cout << "No. iterations: " << epochs << endl;
    cout << "Learning rate: " << learning_rate << endl;
    cout << "Momentum: " << momentum << endl;
    cout << "Epsilon: " << epsilon << endl;
    cout << endl;
    cout << "Training image data: " << training_image_fn << endl;
    cout << "Training label data: " << training_label_fn << endl;
    cout << "No. training sample: " << nTraining << endl
         << endl;
}

void processPerceptron()
{
    cl_int numErr;
    size_t globalWorkSize[1] = {(size_t)(n2 + 1)};
    size_t localWorkSize[1] = {1};
    cl_mem deviceIn2 = clCreateBuffer(context, CL_MEM_READ_WRITE, (n2 + 1) * sizeof(double), NULL, &numErr);
    avalError(context, 13, numErr);
    avalError(context, 14, clSetKernelArg(kernelClearVector, 0, sizeof(cl_mem), &deviceIn2));
    avalError(context, 15, clEnqueueNDRangeKernel(commandQueue, kernelClearVector, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL));

    globalWorkSize[0] = (size_t)(n3 + 1);
    cl_mem deviceIn3 = clCreateBuffer(context, CL_MEM_READ_WRITE, (n3 + 1) * sizeof(double), NULL, &numErr);
    avalError(context, 16, numErr);
    avalError(context, 17, clSetKernelArg(kernelClearVector, 0, sizeof(cl_mem), &deviceIn3));
    avalError(context, 18, clEnqueueNDRangeKernel(commandQueue, kernelClearVector, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL));

    cl_mem deviceOut1 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, (n1 + 1) * sizeof(double), pOut1, &numErr);
    avalError(context, 19, numErr);

    cl_mem deviceW1 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, (n1 + 1) * (n2 + 1) * sizeof(double), pW1, &numErr);
    avalError(context, 21, numErr);

    // int tamanho = (n1 + 1);
    // double *ptr_to_array_data = (double *)malloc(tamanho * sizeof(double));
    // avalError(context, 777, clEnqueueReadBuffer(commandQueue, deviceOut1, CL_TRUE, 0, tamanho * sizeof(double), ptr_to_array_data, 0, NULL, NULL));
    // for (int x = 0; x < tamanho; x++)
    // {
    //         cout << (double)*(ptr_to_array_data + x) << " ";
    // }

    int limit = n2;
    globalWorkSize[0] = (size_t)(n2 + 1);
    for (int i = 1; i <= n1; ++i)
    {
        avalError(context, 22, clSetKernelArg(kernelSumVector, 0, sizeof(cl_mem), &deviceIn2));
        avalError(context, 23, clSetKernelArg(kernelSumVector, 1, sizeof(cl_mem), &deviceOut1));
        avalError(context, 24, clSetKernelArg(kernelSumVector, 2, sizeof(cl_mem), &deviceW1));
        avalError(context, 25, clSetKernelArg(kernelSumVector, 3, sizeof(cl_int), &i));
        avalError(context, 26, clSetKernelArg(kernelSumVector, 4, sizeof(cl_int), &limit));
        avalError(context, 27, clEnqueueNDRangeKernel(commandQueue, kernelSumVector, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL));
    }
    //exit(40);
    // int tamanho = (n2 + 1);
    // double *ptr_to_array_data = (double *)malloc(tamanho * sizeof(double));
    // avalError(context, 777, clEnqueueReadBuffer(commandQueue, deviceIn2, CL_TRUE, 0, tamanho * sizeof(double), ptr_to_array_data, 0, NULL, NULL));
    // for (int x = 0; x < tamanho; x++)
    // {
    //     if (*(ptr_to_array_data + x) != 0)
    //     {
    //         cout << (double)*(ptr_to_array_data + x) << " ";
    //     }
    // }
    // cout << endl;
    // free(ptr_to_array_data);
    // exit(25);

    avalError(context, 28, clReleaseMemObject(deviceOut1));
    avalError(context, 29, clReleaseMemObject(deviceW1));

    cl_mem deviceOut2 = clCreateBuffer(context, CL_MEM_READ_WRITE, (n2 + 1) * sizeof(double), NULL, &numErr);
    avalError(context, 30, numErr);

    globalWorkSize[0] = (size_t)(n2 + 1);
    avalError(context, 31, clSetKernelArg(kernelSigmoid, 0, sizeof(cl_mem), &deviceOut2));
    avalError(context, 32, clSetKernelArg(kernelSigmoid, 1, sizeof(cl_mem), &deviceIn2));
    avalError(context, 33, clEnqueueNDRangeKernel(commandQueue, kernelSigmoid, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL));

    cl_mem deviceW2 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, (n2 + 1) * (n3 + 1) * sizeof(double), pW2, &numErr);
    avalError(context, 34, numErr);

    avalError(context, 35, clReleaseMemObject(deviceIn2));

    avalError(context, 36, clEnqueueReadBuffer(commandQueue, deviceOut2, CL_TRUE, 0, (n2 + 1) * sizeof(double), pOut2, 0, NULL, NULL));

    limit = n3;
    globalWorkSize[0] = (size_t)(n3 + 1);
    for (int i = 1; i <= n2; ++i)
    {
        avalError(context, 37, clSetKernelArg(kernelSumVector, 0, sizeof(cl_mem), &deviceIn3));
        avalError(context, 38, clSetKernelArg(kernelSumVector, 1, sizeof(cl_mem), &deviceOut2));
        avalError(context, 39, clSetKernelArg(kernelSumVector, 2, sizeof(cl_mem), &deviceW2));
        avalError(context, 40, clSetKernelArg(kernelSumVector, 3, sizeof(cl_int), &i));
        avalError(context, 41, clSetKernelArg(kernelSumVector, 4, sizeof(cl_int), &limit));
        avalError(context, 42, clEnqueueNDRangeKernel(commandQueue, kernelSumVector, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL));
    }
    avalError(context, 43, clReleaseMemObject(deviceOut2));
    avalError(context, 44, clReleaseMemObject(deviceW2));

    cl_mem deviceOut3 = clCreateBuffer(context, CL_MEM_READ_WRITE, (n3 + 1) * sizeof(double), NULL, &numErr);
    avalError(context, 45, numErr);

    globalWorkSize[0] = (size_t)(n3 + 1);
    avalError(context, 46, clSetKernelArg(kernelSigmoid, 0, sizeof(cl_mem), &deviceOut3));
    avalError(context, 47, clSetKernelArg(kernelSigmoid, 1, sizeof(cl_mem), &deviceIn3));
    avalError(context, 48, clEnqueueNDRangeKernel(commandQueue, kernelSigmoid, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL));

    avalError(context, 48, clReleaseMemObject(deviceIn3));

    avalError(context, 50, clEnqueueReadBuffer(commandQueue, deviceOut3, CL_TRUE, 0, (n3 + 1) * sizeof(double), pOut3, 0, NULL, NULL));
    avalError(context, 51, clReleaseMemObject(deviceOut3));

    // int tamanho = (n3 + 1);
    // double *ptr_to_array_data = (double *)malloc(tamanho * sizeof(double));
    // avalError(context, 777, clEnqueueReadBuffer(commandQueue, deviceIn3, CL_TRUE, 0, tamanho * sizeof(double), ptr_to_array_data, 0, NULL, NULL));
    // for (int x = 0; x < tamanho; x++)
    // {
    //     if (*(ptr_to_array_data + x) != 0)
    //     {
    //         cout << (double)*(ptr_to_array_data + x) << " ";
    //     }
    // }
    // cout << endl;
    // free(ptr_to_array_data);
}

void initLayersRoundWeight()
{
    // Layer 1 - Layer 2 = Input layer - Hidden layer
    // Initialization for weights Input Layer
    pW1 = (double *)malloc((n1 + 1) * (n2 + 1) * sizeof(double));
    for (int x1 = 1; x1 <= n1; x1++)
    {
        for (int x2 = 1; x2 <= n2; x2++)
        {
            int sign = rand() % 2;
            *(pW1 + (x1 * n2 + x2)) = (double)(rand() % 6) / 10.0;
            if (sign == 1)
            {
                *(pW1 + (x1 * n2 + x2)) = -*(pW1 + (x1 * n2 + x2));
            }
        }
    }

    // Layer 2 - Layer 3 = Hidden layer - Output layer
    // Initialization for weights Input Layer
    pW2 = (double *)malloc((n2 + 1) * (n3 + 1) * sizeof(double));
    for (int x2 = 1; x2 <= n2; x2++)
    {
        for (int x3 = 1; x3 <= n3; x3++)
        {
            int sign = rand() % 2;
            *(pW2 + (x2 * n3 + x3)) = (double)(rand() % 6) / 10.0;
            if (sign == 1)
            {
                *(pW2 + (x2 * n3 + x3)) = -*(pW2 + (x2 * n3 + x3));
            }
        }
    }

    pIn2 = (double *)malloc((n2 + 1) * sizeof(double));
    pOut2 = (double *)malloc((n2 + 1) * sizeof(double));

    // Layer 3 - Output layer
    pIn3 = (double *)malloc((n3 + 1) * sizeof(double));
    pOut3 = (double *)malloc((n3 + 1) * sizeof(double));

    // cout << "**************2 " << endl;
    // for (int x1 = 1; x1 <= n2; x1++)
    // {
    //     for (int x2 = 1; x2 <= n3; x2++)
    //     {
    //         cout << *(w1Temp2 + (x1 * n3 + x2)) << " ";
    //     }
    //     cout << endl;
    // }
    // cout << endl;
}

void saveMLP(string file_name)
{
    ofstream file(file_name.c_str(), ios::out);
    // Input layer - Hidden layer
    for (int i = 1; i <= n1; ++i)
    {
        for (int j = 1; j <= n2; ++j)
        {
            file << *(pW1 + i * n2 + j) << " ";
        }
        file << endl;
    }
    // Hidden layer - Output layer
    for (int i = 1; i <= n2; ++i)
    {
        for (int j = 1; j <= n3; ++j)
        {
            file << *(pW2 + i * n3 + j) << " ";
        }
        file << endl;
    }
    file.close();
}

double squareError()
{
    double res = 0.0;
    for (int i = 1; i <= n3; ++i)
    {
        res += (*(pOut3 + i) - *(pExpected + i)) * (*(pOut3 + i) - *(pExpected + i));
    }
    res *= 0.5;
    return res;
}

void backPropagation()
{
    double sum;
    double *pTheta3 = (double *)malloc((n3 + 1) * sizeof(double));
    for (int i = 1; i <= n3; ++i)
    {
        *(pTheta3 + i) = *(pOut3 + i) * (1 - *(pOut3 + i)) * (*(pExpected + i) - *(pOut3 + i));
    }
    double *pTheta2 = (double *)malloc((n2 + 1) * sizeof(double));
    for (int i = 1; i <= n2; ++i)
    {
        sum = 0.0;
        for (int j = 1; j <= n3; ++j)
        {
            sum += *(pW2 + i * n3 + j) * *(pTheta3 + j);
        }
        *(pTheta2 + i) = *(pOut2 + i) * (1 - *(pOut2 + i)) * sum;
    }
    for (int i = 1; i <= n2; ++i)
    {
        for (int j = 1; j <= n3; ++j)
        {
            *(pDelta2 + i * n3 + j) = (learning_rate * *(pTheta3 + j) * *(pOut2 + i)) + (momentum * *(pDelta2 + i * n3 + j));
            *(pW2 + i * n3 + j) += *(pDelta2 + i * n3 + j);
        }
    }

    for (int i = 1; i <= n1; ++i)
    {
        for (int j = 1; j <= n2; j++)
        {
            *(pDelta1 + i * n2 + j) = (learning_rate * *(pTheta2 + j) * *(pOut1 + i)) + (momentum * *(pDelta1 + i * n2 + j));
            *(pW1 + i * n2 + j) += *(pDelta1 + i * n2 + j);
        }
    }
    free(pTheta3);
    free(pTheta2);
}

int learning()
{
    int numErr;
    free(pDelta1);
    free(pDelta2);

    pDelta1 = (double *)malloc((n1 + 1) * (n2 + 1) * sizeof(double));
    pDelta2 = (double *)malloc((n2 + 1) * (n3 + 1) * sizeof(double));

    // zerar deviceDelta1
    size_t globalWorkSize[1] = {(size_t)(n1 + 1) * (n2 + 1)};
    size_t localWorkSize[1] = {1};
    cl_mem deviceDelta1 = clCreateBuffer(context, CL_MEM_READ_WRITE, (n1 + 1) * (n2 + 1) * sizeof(double), NULL, &numErr);
    avalError(context, 5, numErr);
    avalError(context, 6, clSetKernelArg(kernelClearVector, 0, sizeof(cl_mem), &deviceDelta1));
    avalError(context, 7, clEnqueueNDRangeKernel(commandQueue, kernelClearVector, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL));
    avalError(context, 8, clEnqueueReadBuffer(commandQueue, deviceDelta1, CL_TRUE, 0, (n1 + 1) * (n2 + 1) * sizeof(double), pDelta1, 0, NULL, NULL));
    avalError(context, 9, clReleaseMemObject(deviceDelta1));

    globalWorkSize[0] = (size_t)(n2 + 1) * (n3 + 1);
    cl_mem deviceDelta2 = clCreateBuffer(context, CL_MEM_READ_WRITE, (n2 + 1) * (n3 + 1) * sizeof(double), NULL, &numErr);
    avalError(context, 10, numErr);
    avalError(context, 11, clSetKernelArg(kernelClearVector, 0, sizeof(cl_mem), &deviceDelta2));
    avalError(context, 12, clEnqueueNDRangeKernel(commandQueue, kernelClearVector, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL));
    avalError(context, 13, clEnqueueReadBuffer(commandQueue, deviceDelta2, CL_TRUE, 0, (n2 + 1) * (n3 + 1) * sizeof(double), pDelta2, 0, NULL, NULL));
    clReleaseMemObject(deviceDelta2);
    for (int i = 1; i <= epochs; ++i)
    {
        processPerceptron();
        backPropagation();
        if (squareError() < epsilon)
        {
            return i;
        }
    }
    return epochs;
}
void input()
{
    char number;
    if (pOut1 == 0)
    {
        free(pOut1);
    }
    pOut1 = (double *)malloc((n1 + 1) * sizeof(double));
    *pOut1 = 0;
    for (int h = 1; h <= height; h++)
    {
        for (int w = 1; w <= width; w++)
        {
            image.read(&number, sizeof(char));
            if (number == 0)
            {
                *(pOut1 + (h - 1) * width + w) = 0;
            }
            else
            {
                *(pOut1 + (h - 1) * width + w) = 1;
            }
        }
    }
    // cout << "Image Ponteiro:" << endl;
    // for (int h = 0; h < height; h++)
    // {
    //     for (int w = 1; w <= width; w++)
    //     {
    //         cout << *(pOut1 + (h * width + w));
    //     }
    //     cout << endl;
    // }

    label.read(&number, sizeof(char));
    if (pExpected == 0)
    {
        free(pExpected);
    }
    pExpected = (double *)malloc((n3 + 1) * sizeof(double));
    for (int i = 1; i <= n3; i++)
    {
        *(pExpected + i) = 0;
    }
    *(pExpected + number + 1) = 1.0;
    cout << "Label: " << (int)(number) << endl;
}

void training()
{
    for (int sample = 1; sample <= nTraining; ++sample)
    {
        cout << "Sample " << sample << endl;
        // Getting (image, label)
        input();
        // Learning process: Perceptron (Forward procedure) - Back propagation
        int nIterations = learning();
        // Write down the squared error
        cout << "No. iterations: " << nIterations << endl;
        printf("Error: %0.6lf\n\n", squareError());
        report << "Sample " << sample << ": No. iterations = " << nIterations << ", Error = " << squareError() << endl;
    }
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
            clReleaseContext(context);
            exit(pos);
        case 4:
            cout << "initKernel() - Failed to create kernelClearVector!" << endl;
            cleanKernels();
            cleanBuffers();
            cleanOpenCL();
            exit(pos);
        case 5:
            cout << "Error setting kernel arguments [" << errNum << "]" << endl;
            cleanKernels();
            cleanBuffers();
            cleanOpenCL();
            exit(pos);
        case 6:
            cout << "Error queuing kernel for execution. Errnum [" << errNum << "]" << endl;
            cleanKernels();
            cleanBuffers();
            cleanOpenCL();
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
    program = createProgram(context, *device, "mlp.cl");
    avalError(program, 3, CL_SUCCESS);
    deviceGlobal = device;
    initCommandQueue();
}
void initCommandQueue()
{
    if (commandQueue != 0)
    {
        cout << "Liberando commandQueue" << endl;
        clReleaseCommandQueue(commandQueue);
    }
    cout << "Device Global " << *deviceGlobal << endl;
    commandQueue = createCommandQueue(context, deviceGlobal);
    avalError(commandQueue, 2, CL_SUCCESS);
}

void initKernels()
{
    kernelClearVector = clCreateKernel(program, "clearVector", NULL);
    avalError(kernelClearVector, 400, CL_SUCCESS);
    kernelSumVector = clCreateKernel(program, "sumVector", NULL);
    avalError(kernelClearVector, 401, CL_SUCCESS);
    kernelSigmoid = clCreateKernel(program, "sigmoid", NULL);
    avalError(kernelSigmoid, 402, CL_SUCCESS);
}

void cleanOpenCL()
{
    clReleaseProgram(program);
    if (commandQueue != 0)
    {
        clReleaseCommandQueue(commandQueue);
    }
    clReleaseContext(context);
}
void cleanBuffers()
{
    free(pW1);
    free(pDelta1);
    free(pOut1);
    free(pW2);
    free(pDelta2);
    free(pIn2);
    free(pOut2);
    free(pTheta2);
    free(pIn3);
    free(pOut3);
    free(pTheta3);
    free(pExpected);
}

void cleanKernels()
{
    if (kernelClearVector != 0)
        clReleaseKernel(kernelClearVector);
    if (kernelSumVector != 0)
        clReleaseKernel(kernelSumVector);
    if (kernelSigmoid != 0)
        clReleaseKernel(kernelSigmoid);
}
