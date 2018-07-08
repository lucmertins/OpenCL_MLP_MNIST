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
double *w1[n1 + 1], *delta1[n1 + 1], *out1;
cl_mem deviceW1 = 0, deviceDelta1 = 0, deviceOut1 = 0;

// hidden layer
double *w2[n2 + 1], *delta2[n2 + 1], *in2, *out2, *theta2;
cl_mem deviceW2 = 0, deviceDelta2 = 0, deviceIn2 = 0, deviceOut2 = 0, deviceTheta2 = 0;

// Output layer
double *in3, *out3, *theta3;
cl_mem deviceIn3 = 0, deviceOut3 = 0, deviceTheta3 = 0;

double expected[n3 + 1];
cl_mem deviceExpected = 0;
// Numero de exemplos
const int nTraining = 60000;

cl_context context = 0;
cl_command_queue commandQueue = 0;
cl_program program = 0;
cl_kernel kernelClearVector = 0;

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

void initLayersRoundWeight()
{
    // size_t globalWorkSize[1] = {(size_t)((n1 + 1) * (n2 + 1))};
    // size_t localWorkSize[1] = {1};

    // Initialization for weights Input Layer
    double *w1Temp = (double *)malloc((n1 + 1) * (n2 + 1) * sizeof(double));
    for (int x1 = 1; x1 <= n1; x1++)
    {
        for (int x2 = 1; x2 <= n2; x2++)
        {
            int sign = rand() % 2;
            *(w1Temp + (x1 * n2 + x2)) = (double)(rand() % 6) / 10.0;
            if (sign == 1)
            {
                *(w1Temp + (x1 * n2 + x2)) = -*(w1Temp + (x1 * n2 + x2));
            }
        }
    }

    // Layer 1 - Layer 2 = Input layer - Hidden layer
    cl_mem deviceW1 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, (n1 + 1) * (n2 + 1) * sizeof(double), w1Temp, NULL);
    cl_mem deviceDelta1 = clCreateBuffer(context, CL_MEM_READ_WRITE, (n1 + 1) * (n2 + 1) * sizeof(double), NULL, NULL);
    free(w1Temp);

    // deviceOut1 foi criado na função Input
    // out1 = new double[n1 + 1];

    // Initialization for weights Input Layer
    double *w2Temp = (double *)malloc((n2 + 1) * (n3 + 1) * sizeof(double));
    for (int x2 = 1; x2 <= n2; x2++)
    {
        for (int x3 = 1; x3 <= n3; x3++)
        {
            int sign = rand() % 2;
            *(w2Temp + (x2 * n3 + x3)) = (double)(rand() % 6) / 10.0;
            if (sign == 1)
            {
                *(w2Temp + (x2 * n3 + x3)) = -*(w2Temp + (x2 * n3 + x3));
            }
        }
    }
    // Layer 2 - Layer 3 = Hidden layer - Output layer
    cl_mem deviceW2 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, (n2 + 1) * (n3 + 1) * sizeof(double), w2Temp, NULL);
    cl_mem deviceDelta2 = clCreateBuffer(context, CL_MEM_READ_WRITE, (n2 + 1) * (n3 + 1) * sizeof(double), NULL, NULL);
    free(w2Temp);

    cl_mem deviceIn2 = clCreateBuffer(context, CL_MEM_READ_WRITE, (n2 + 1) * sizeof(double), NULL, NULL);
    cl_mem deviceOut2 = clCreateBuffer(context, CL_MEM_READ_WRITE, (n2 + 1) * sizeof(double), NULL, NULL);
    cl_mem deviceTheta2 = clCreateBuffer(context, CL_MEM_READ_WRITE, (n2 + 1) * sizeof(double), NULL, NULL);

    // Layer 3 - Output layer
    cl_mem deviceIn3 = clCreateBuffer(context, CL_MEM_READ_WRITE, (n3 + 1) * sizeof(double), NULL, NULL);
    cl_mem deviceOut3 = clCreateBuffer(context, CL_MEM_READ_WRITE, (n3 + 1) * sizeof(double), NULL, NULL);
    cl_mem deviceTheta3 = clCreateBuffer(context, CL_MEM_READ_WRITE, (n3 + 1) * sizeof(double), NULL, NULL);

    // double *w1Temp2 = (double *)malloc((n2 + 1) * (n3 + 1) * sizeof(double));
    // clEnqueueReadBuffer(commandQueue, deviceW2, CL_TRUE, 0, (n2 + 1) * (n3 + 1) * sizeof(double), w1Temp2, 0, NULL, NULL);

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
    // free(w1Temp2);
}

void saveMLP(string file_name)
{
    ofstream file(file_name.c_str(), ios::out);
    // Input layer - Hidden layer
    for (int i = 1; i <= n1; ++i)
    {
        for (int j = 1; j <= n2; ++j)
        {
            file << w1[i][j] << " ";
        }
        file << endl;
    }
    // Hidden layer - Output layer
    for (int i = 1; i <= n2; ++i)
    {
        for (int j = 1; j <= n3; ++j)
        {
            file << w2[i][j] << " ";
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
        res += (out3[i] - expected[i]) * (out3[i] - expected[i]);
    }
    res *= 0.5;
    return res;
}

void backPropagation()
{
    double sum;

    for (int i = 1; i <= n3; ++i)
    {
        theta3[i] = out3[i] * (1 - out3[i]) * (expected[i] - out3[i]);
    }

    for (int i = 1; i <= n2; ++i)
    {
        sum = 0.0;
        for (int j = 1; j <= n3; ++j)
        {
            sum += w2[i][j] * theta3[j];
        }
        theta2[i] = out2[i] * (1 - out2[i]) * sum;
    }

    for (int i = 1; i <= n2; ++i)
    {
        for (int j = 1; j <= n3; ++j)
        {
            delta2[i][j] = (learning_rate * theta3[j] * out2[i]) + (momentum * delta2[i][j]);
            w2[i][j] += delta2[i][j];
        }
    }

    for (int i = 1; i <= n1; ++i)
    {
        for (int j = 1; j <= n2; j++)
        {
            delta1[i][j] = (learning_rate * theta2[j] * out1[i]) + (momentum * delta1[i][j]);
            w1[i][j] += delta1[i][j];
        }
    }
}

int learning()
{
    // zerar deviceDelta1
    size_t globalWorkSize[1] = {10};
    size_t localWorkSize[1] = {1};
    avalError(context, 5, clSetKernelArg(kernelClearVector, 0, sizeof(cl_mem), &deviceDelta1));
    initCommandQueue();
    avalError(context, 6, clEnqueueNDRangeKernel(commandQueue, kernelClearVector, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL));
    cout<<"learning "<<endl;
    avalError(context, 7, clFinish(commandQueue));
    double *temp = (double *)malloc((n1 + 1) * (n2 + 1) * sizeof(double));
    avalError(context, 100, clEnqueueReadBuffer(commandQueue, deviceDelta1, CL_TRUE, 0, (n1) * (n2) * sizeof(double), temp, 0, NULL, NULL));

    // cout << endl
    //      << "Buffer clean" << endl;
    // for (int x = 0; x < (n1 + 1) * (n2 + 1); x++)
    // {
    //     if (*(temp + x) != 0.0)
    //     {
    //         cout << x << "     " << *(temp + x) << endl;
    //     }
    // }
    // cout << endl;
    // free(temp);

    // for (int i = 1; i <= n2; ++i)
    // {
    //     for (int j = 1; j <= n3; ++j)
    //     {
    //         delta2[i][j] = 0.0;
    //     }
    // }
    // for (int i = 1; i <= epochs; ++i)
    // {
    //     processPerceptron();
    //     backPropagation();
    //     if (squareError() < epsilon)
    //     {
    //         return i;
    //     }
    // }
    return epochs;
}
void input()
{
    // Reading image
    char number;
    // mudar para uso com ponteiros.

    double *img = (double *)malloc((width * height) * sizeof(double));
    for (int h = 0; h < height; h++)
    {
        for (int w = 0; w < width; w++)
        {
            image.read(&number, sizeof(char));
            if (number == 0)
            {
                *(img + (h * width + w)) = 0;
            }
            else
            {
                *(img + (h * width + w)) = 1;
            }
        }
    }
    // cout << "Image Ponteiro:" << endl;
    // for (int h = 0; h < height; h++)
    // {
    //     for (int w = 0; w < width; w++)
    //     {
    //         cout << *(img + (h * width + w));
    //     }
    //     cout << endl;
    // }
    if (deviceOut1 != 0)
    {
        clReleaseMemObject(deviceOut1);
    }
    deviceOut1 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, (n1 + 1) * sizeof(double), img, NULL);
    free(img);
    // double *imgTemp = (double *)malloc(width * height * sizeof(double));
    // clEnqueueReadBuffer(commandQueue, deviceOut1, CL_TRUE, 0, (n1 + 1) * sizeof(double), imgTemp, 0, NULL, NULL);

    // cout << "Imagem Buffer read" << endl;
    // for (int h = 0; h < height; h++)
    // {
    //     for (int w = 0; w < width; w++)
    //     {
    //         cout << *(imgTemp + (h * width + w));
    //     }
    //     cout << endl;
    // }
    // cout << endl;
    // free(imgTemp);

    // Reading label
    label.read(&number, sizeof(char));

    double *expectedTemp = (double *)malloc((n3 + 1) * sizeof(double));
    for (int i = 1; i <= n3; i++)
    {
        *(expectedTemp + i) = 0;
    }
    *(expectedTemp + number + 1) = 1.0;
    if (deviceExpected != 0)
    {
        clReleaseMemObject(deviceExpected);
    }
    deviceExpected = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, (n3 + 1) * sizeof(double), expectedTemp, NULL);
    free(expectedTemp);
    cout << "Label: " << (int)(number) << endl;
}

void training()
{
    for (int sample = 1; sample <= 2; ++sample)
    {
        cout << "Sample " << sample << endl;

        // Getting (image, label)
        input();

        // // Learning process: Perceptron (Forward procedure) - Back propagation
        int nIterations = learning();

        // // Write down the squared error
        // cout << "No. iterations: " << nIterations << endl;
        // printf("Error: %0.6lf\n\n", squareError());
        // report << "Sample " << sample << ": No. iterations = " << nIterations << ", Error = " << squareError() << endl;
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
}
void initCommandQueue()
{
    if (commandQueue != 0)
    {
        cout<<"Liberando commandQueue"<<endl;
        clReleaseCommandQueue(commandQueue);
    }
    cout << "Device Global " << *deviceGlobal << endl;
    commandQueue = createCommandQueue(context, deviceGlobal);
    avalError(commandQueue, 2, CL_SUCCESS);
}

void initKernels()
{
    kernelClearVector = clCreateKernel(program, "clearVector", NULL);
    avalError(kernelClearVector, 4, CL_SUCCESS);
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
    if (deviceW1 != 0)
        clReleaseMemObject(deviceW1);
    if (deviceDelta1 != 0)
        clReleaseMemObject(deviceDelta1);
    if (deviceOut1 != 0)
        clReleaseMemObject(deviceOut1);
    if (deviceW2 != 0)
        clReleaseMemObject(deviceW2);
    if (deviceDelta2 != 0)
        clReleaseMemObject(deviceDelta2);
    if (deviceIn2 != 0)
        clReleaseMemObject(deviceIn2);
    if (deviceOut2 != 0)
        clReleaseMemObject(deviceOut2);
    if (deviceTheta2 != 0)
        clReleaseMemObject(deviceTheta2);
    if (deviceIn3 != 0)
        clReleaseMemObject(deviceIn3);
    if (deviceOut3 != 0)
        clReleaseMemObject(deviceOut3);
    if (deviceTheta3 != 0)
        clReleaseMemObject(deviceTheta3);
    if (deviceExpected != 0)
        clReleaseMemObject(deviceExpected);
}

void cleanKernels()
{
    if (kernelClearVector != 0)
        clReleaseKernel(kernelClearVector);
}
