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

using namespace std;

ifstream image;
ifstream label;
ofstream report;

void showDate()
{
    char result[100];
    time_t t = time(NULL);
    strftime(result, sizeof(result), "%d/%m/%Y %H:%M", localtime(&t));
    report << "Tempo:  " << result << endl;
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
    initKernels();

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

    cleanBuffers();

    report.close();
    image.close();
    label.close();
}
