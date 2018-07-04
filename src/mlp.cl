
__kernel void clearN1l(__global const float *n1)
{
    int base = 4*get_global_id(0);
    n1[base++]=0;
    n1[base++]=0;
    n1[base++]=0;
    n1[base++]=0;

}
