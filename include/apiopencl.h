///
//  Constants
//
// 0 pc casa 
// 2 notebook com bumblebee
const int PLATAFORM_SELECT = 0;    
const int ARRAY_SIZE = 1000;

cl_context CreateContext();
cl_command_queue CreateCommandQueue(cl_context context, cl_device_id *device);
cl_program CreateProgram(cl_context context, cl_device_id device, const char *fileName);
bool CreateMemObjects(cl_context context, cl_mem memObjects[3], float *a, float *b);
void Cleanup(cl_context context, cl_command_queue commandQueue, cl_program program, cl_kernel kernel, cl_mem memObjects[3]);
