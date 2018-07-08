
__kernel void clearVector(__global double *v)
{
    int gid = get_global_id(0);
    printf("before %d ",gid);
    v[gid]=0;
    //printf(" after %d \n",gid);

}