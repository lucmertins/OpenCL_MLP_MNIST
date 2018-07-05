#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

void showPlataforms();
void showDevices(int plataformSelect);
cl_context createContext(int plataformSelect);
cl_command_queue createCommandQueue(cl_context context, cl_device_id *device);
cl_program createProgram(cl_context context, cl_device_id device, const char *fileName);
