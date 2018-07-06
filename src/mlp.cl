
__kernel void clearVector(__global double *v)
{
    int gid = get_global_id(0);
    v[gid]=0;

}