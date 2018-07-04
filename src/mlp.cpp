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
const int n2 = 128; // hidden layer
const int n3 = 10; // Ten classes: 0 - 9
const int epochs = 512;
const double learning_rate = 1e-3;
const double momentum = 0.9;
const double epsilon = 1e-3;

// MLP Definition
// input layer
double *w1[n1 + 1],*delta1[n1 + 1], *out1;

// hidden layer
double *w2[n2 + 1],*delta2[n2 + 1], *in2, *out2, *theta2;

// Output layer
double *in3, *out3,*theta3;

double expected[n3 + 1];


ifstream image;
ifstream label;
ofstream report;


double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

void processPerceptron() {
    for (int i = 1; i <= n2; ++i) {
		in2[i] = 0.0;
	}

    for (int i = 1; i <= n3; ++i) {
		in3[i] = 0.0;
	}

    for (int i = 1; i <= n1; ++i) {
        for (int j = 1; j <= n2; ++j) {
            in2[j] += out1[i] * w1[i][j];
		}
	}

    for (int i = 1; i <= n2; ++i) {
		out2[i] = sigmoid(in2[i]);
	}

    for (int i = 1; i <= n2; ++i) {
        for (int j = 1; j <= n3; ++j) {
            in3[j] += out2[i] * w2[i][j];
		}
	}

    for (int i = 1; i <= n3; ++i) {
		out3[i] = sigmoid(in3[i]);
	}
}

void showDate(){
    char result[100];
    time_t t= time(NULL);
    strftime(result, sizeof(result), "%d/%m/%Y %H:%M", localtime(&t));
    report << "Tempo:  " << result << endl;
}

int main(int argc, char *argv[])
{
    aboutTraining();
    report.open(report_fn.c_str(), ios::out);
    image.open(training_image_fn.c_str(), ios::in | ios::binary); // Binary image file
    label.open(training_label_fn.c_str(), ios::in | ios::binary); // Binary label file

// Retirar headers do arquivo
    char number;
    for (int i = 1; i <= 16; ++i)
    {
        image.read(&number, sizeof(char));
    }
    for (int i = 1; i <= 8; ++i)
    {
        label.read(&number, sizeof(char));
    }
    initLayersRoundWeight();

    showDate();

    training();
    saveMLP(model_fn);

    showDate();
    report.close();
    image.close();
    label.close();
    

}
