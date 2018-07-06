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
cl_mem deviceW1, deviceDelta1, deviceOut1;

// hidden layer
double *w2[n2 + 1], *delta2[n2 + 1], *in2, *out2, *theta2;
cl_mem deviceW2, deviceDelta2, deviceIn2, deviceOut2, deviceTheta2;

// Output layer
double *in3, *out3, *theta3;
cl_mem deviceIn3, deviceOut3, deviceTheta3;

double expected[n3 + 1];
// Numero de exemplos
const int nTraining = 60000;

cl_context context = 0;
cl_command_queue commandQueue = 0;
cl_program program = 0;
//cl_kernel kernelClearN2 = 0, kernelIncrementN2 = 0;

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

void clearBuffers()
{
    clReleaseMemObject(deviceW1);
    clReleaseMemObject(deviceDelta1);
    clReleaseMemObject(deviceOut1);
    clReleaseMemObject(deviceW2);
    clReleaseMemObject(deviceDelta2);
    clReleaseMemObject(deviceIn2);
    clReleaseMemObject(deviceOut2);
    clReleaseMemObject(deviceTheta2);
    clReleaseMemObject(deviceIn3);
    clReleaseMemObject(deviceOut3);
    clReleaseMemObject(deviceTheta3);
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

    cl_mem deviceOut1 = clCreateBuffer(context, CL_MEM_READ_WRITE, (n1 + 1) * sizeof(double), NULL, NULL);
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
    for (int i = 1; i <= n1; ++i)
    {
        for (int j = 1; j <= n2; ++j)
        {
            delta1[i][j] = 0.0;
        }
    }
    for (int i = 1; i <= n2; ++i)
    {
        for (int j = 1; j <= n3; ++j)
        {
            delta2[i][j] = 0.0;
        }
    }
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
    // Reading image
    char number;
// mudar para uso com ponteiros.

    int d[width + 1][height + 1];
    for (int j = 1; j <= height; ++j)
    {
        for (int i = 1; i <= width; ++i)
        {
            image.read(&number, sizeof(char));
            if (number == 0)
            {
                d[i][j] = 0;
            }
            else
            {
                d[i][j] = 1;
            }
        }
    }
    /*	
	cout << "Image:" << endl;
	for (int j = 1; j <= height; ++j) {
		for (int i = 1; i <= width; ++i) {
			cout << d[i][j];
		}
		cout << endl;
	}
*/

// copiar dados para deviceOut1
    // for (int j = 1; j <= height; ++j)
    // {
    //     for (int i = 1; i <= width; ++i)
    //     {
    //         int pos = i + (j - 1) * width;
    //         out1[pos] = d[i][j];
    //     }
    // }

    // Reading label
    label.read(&number, sizeof(char));
    for (int i = 1; i <= n3; ++i)
    {
        expected[i] = 0.0;
    }
    expected[number + 1] = 1.0;

    cout << "Label: " << (int)(number) << endl;
}

void training()
{
    for (int sample = 1; sample <= nTraining; ++sample)
    {
        cout << "Sample " << sample << endl;

        // Getting (image, label)
        input();

        // // Learning process: Perceptron (Forward procedure) - Back propagation
        // int nIterations = learning();

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
    program = createProgram(context, *device, "mlp.cl");
    avalError(program, 3, CL_SUCCESS);
}

void cleanOpenCL()
{
    clReleaseProgram(program);
    clReleaseCommandQueue(commandQueue);
    clReleaseContext(context);
}
