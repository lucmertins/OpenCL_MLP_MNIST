
__kernel void sigmoid(__global double *out, __global double *in)
{
    int i = get_global_id(0);
    out[i]=1.0 / (1.0 + exp(-in[i]));
}

__kernel void clearVector(__global double *v)
{
    int gid = get_global_id(0);
    v[gid]=0;
}

__kernel void sumVector(__global double *in, __global double *out, __global double *w, const int i, const int n)
{
    int j = get_global_id(0);  //128
//    if (j!=0) {
    in[j] += out[i] * *(w + i*n + j);
    //if (j==1){
    //  printf("f %f ",in[j]);
    //}
//    }
}

__kernel void multiMatrix(__global double *in, __global double *out, __global double *w, const int n1, const int n2)
{
    int j = get_global_id(0); //129
    if (j!=0){
        double result=0;
        for (int i = 1; i <= n1; i++) 
        {
            result += *(out+i) * *(w + i *n2 + j);
            // if (j==53){
            //     printf("[%f %f]",*(out+i),*(w + j * limit + i));
            // }
        }
        in[j]=1.0 / (1.0 + exp(-result));
    }
}