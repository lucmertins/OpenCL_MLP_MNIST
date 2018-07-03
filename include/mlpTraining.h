#include <string>

using namespace std;

// Training image 
const string training_image_fn = "../mnist/train-images.idx3-ubyte";

// Training label 
const string training_label_fn = "../mnist/train-labels.idx1-ubyte";

// Numero de exemplos
const int nTraining = 60000;

// Relatório
const string report_fn = "relatorioTreinamento.log";

const string model_fn = "mlp.obj";

extern double expected[];
extern ifstream image;
extern ifstream label;
extern ofstream report;


void aboutTraining();
void init_mlpTraining();
void training();
void write_matrix(string file_name);