#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>

#include "mlp.h"
#include "mlpTraining.h"


using namespace std;



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

void initLayersRoundWeight()
{
    // Layer 1 - Layer 2 = Input layer - Hidden layer
    for (int i = 1; i <= n1; ++i)
    {
        w1[i] = new double[n2 + 1];
        delta1[i] = new double [n2 + 1];
    }

    out1 = new double[n1 + 1];

    // Layer 2 - Layer 3 = Hidden layer - Output layer
    for (int i = 1; i <= n2; ++i)
    {
        w2[i] = new double[n3 + 1];
        delta2[i] = new double [n3 + 1];
    }

    in2 = new double[n2 + 1];
    out2 = new double[n2 + 1];
    theta2 = new double [n2 + 1];

    // Layer 3 - Output layer
    in3 = new double[n3 + 1];
    out3 = new double[n3 + 1];
    theta3 = new double [n3 + 1];
    // Initialization for weights from Input layer to Hidden layer
    for (int i = 1; i <= n1; ++i) {
        for (int j = 1; j <= n2; ++j) {
            int sign = rand() % 2;
            w1[i][j] = (double)(rand() % 6) / 10.0;
            if (sign == 1) {
				w1[i][j] = - w1[i][j];
			}
        }
	}
    // Initialization for weights from Hidden layer to Output layer
    for (int i = 1; i <= n2; ++i) {
        for (int j = 1; j <= n3; ++j) {
            int sign = rand() % 2;
            w2[i][j] = (double)(rand() % 10 + 1) / (10.0 * n3);
            if (sign == 1) {
				w2[i][j] = - w2[i][j];
			}
        }
	}
}

void saveMLP(string file_name) {
    ofstream file(file_name.c_str(), ios::out);
	// Input layer - Hidden layer
    for (int i = 1; i <= n1; ++i) {
        for (int j = 1; j <= n2; ++j) {
			file << w1[i][j] << " ";
		}
		file << endl;
    }
	// Hidden layer - Output layer
    for (int i = 1; i <= n2; ++i) {
        for (int j = 1; j <= n3; ++j) {
			file << w2[i][j] << " ";
		}
        file << endl;
    }
	file.close();
}

double squareError(){
    double res = 0.0;
    for (int i = 1; i <= n3; ++i) {
        res += (out3[i] - expected[i]) * (out3[i] - expected[i]);
	}
    res *= 0.5;
    return res;
}

void backPropagation() {
    double sum;

    for (int i = 1; i <= n3; ++i) {
        theta3[i] = out3[i] * (1 - out3[i]) * (expected[i] - out3[i]);
	}

    for (int i = 1; i <= n2; ++i) {
        sum = 0.0;
        for (int j = 1; j <= n3; ++j) {
            sum += w2[i][j] * theta3[j];
		}
        theta2[i] = out2[i] * (1 - out2[i]) * sum;
    }

    for (int i = 1; i <= n2; ++i) {
        for (int j = 1; j <= n3; ++j) {
            delta2[i][j] = (learning_rate * theta3[j] * out2[i]) + (momentum * delta2[i][j]);
            w2[i][j] += delta2[i][j];
        }
	}

    for (int i = 1; i <= n1; ++i) {
        for (int j = 1 ; j <= n2 ; j++ ) {
            delta1[i][j] = (learning_rate * theta2[j] * out1[i]) + (momentum * delta1[i][j]);
            w1[i][j] += delta1[i][j];
        }
	}
}

int learning() {
    for (int i = 1; i <= n1; ++i) {
        for (int j = 1; j <= n2; ++j) {
			delta1[i][j] = 0.0;
		}
	}
    for (int i = 1; i <= n2; ++i) {
        for (int j = 1; j <= n3; ++j) {
			delta2[i][j] = 0.0;
		}
	}
    for (int i = 1; i <= epochs; ++i) {
        processPerceptron();
        backPropagation();
        if (squareError() < epsilon) {
			return i;
		}
    }
    return epochs;
}
void input() {
	// Reading image
    char number;

    int d[width + 1][height + 1];
    for (int j = 1; j <= height; ++j) {
        for (int i = 1; i <= width; ++i) {
            image.read(&number, sizeof(char));
            if (number == 0) {
				d[i][j] = 0; 
			} else {
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
    for (int j = 1; j <= height; ++j) {
        for (int i = 1; i <= width; ++i) {
            int pos = i + (j - 1) * width;
            out1[pos] = d[i][j];
        }
	}

	// Reading label
    label.read(&number, sizeof(char));
    for (int i = 1; i <= n3; ++i) {
		expected[i] = 0.0;
	}
    expected[number + 1] = 1.0;
    
    cout << "Label: " << (int)(number) << endl;
}

void training(){
    for (int sample = 1; sample <= nTraining; ++sample) {
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
