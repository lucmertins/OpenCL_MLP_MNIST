
__kernel void clearN2(__global double *n2)
{
    int base = get_global_id(0);
    n2[base]=0;

}

__kernel void incrementN2(__global double *n2)
{
    int base =get_global_id(0);
    n2[base]=4;
//    printf("%d ",base);
}
