
__kernel void clearN2(__global double *n2)
{
    int base = get_global_id(0);
    n2[base]=0;

}

__kernel void incrementN2(__global double *n2)
{
    int base =get_global_id(0);
    if (base>0){
        n2[base]=n2[base-1]+1;
    }
//    printf("%d ",base);
}

__kernel matriz2D(__global int * array, __global int width,__global int height) {
    int id = get_global_id(0);
    int our_value = array[id];
    int x = id % width; //This will depend on how the memory is laid out in the 2d array. 
    int y = id / width; //If it's not row-major, then you'll need to flip these two statements.
    *array=x;
}
