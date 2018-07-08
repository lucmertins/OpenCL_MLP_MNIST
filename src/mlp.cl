
__kernel void clearVector(__global double *v)
{
    int gid = get_global_id(0);
    v[gid]=0;
}

__kernel void sumVector(__global double *in, __global double *out, __global double *w, const int i, const int n) {
    int j = get_global_id(0);
//    if (j!=0) {
        in[j] += out[i] * *(w + i*n + j);
        //if (j==1){
          //  printf("f %f ",in[j]);
        //}
//    }
}
__kernel void matrixVectorMul(__global float* resultVector,    __global float* matrixA,    __global float* vectorB,     int width_A)
{
    int tx = get_global_id(0); 

    float value = 0;
    for (unsigned int k = 0; k < width_A; ++k) {
        value += matrixA[tx * width_A + k] * vectorB[k];
    }

    resultVector[tx] = value;
}

__kernel void sigmoid(__global double *out, __global double *in) {
    int i = get_global_id(0);
//    if (i!=0) {
        out[i]=1.0 / (1.0 + exp(-in[i]));
//    }
}
