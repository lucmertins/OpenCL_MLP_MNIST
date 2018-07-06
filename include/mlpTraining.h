#include <string>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

using namespace std;

// Training image
const string training_image_fn = "../mnist/train-images.idx3-ubyte";

// Training label
const string training_label_fn = "../mnist/train-labels.idx1-ubyte";

// Relatório
const string report_fn = "relatorioTreinamento.log";

const string model_fn = "mlp.obj";

extern ifstream image;
extern ifstream label;
extern ofstream report;

void aboutTraining();
void initLayersRoundWeight();
void training();
void saveMLP(string file_name);

void initOpenCL(int plataformId, cl_device_id *device);
void clearBuffers();
void cleanOpenCL();