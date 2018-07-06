
__kernel void clearN2(__global double *n2)
{
    int base = get_global_id(0);
    n2[base]=0;

}

__kernel void incrementN2(__global double *n2)
{
    int base =get_global_id(0);
    n2[base]=base;
    // if (base>0){
    //     n2[base]=n2[base-1]+1;
    // }
    // printf(" %d ",base);
}

__kernel void matriz2D(__global int * array, const int width,const int height) {
    int idG = get_global_id(0);
    int idL = get_local_id(0);
    int our_value = array[idG];
    int x = idG % width; //This will depend on how the memory is laid out in the 2d array. 
    int y = idG / width; //If it's not row-major, then you'll need to flip these two statements.

    for (int w=0;w<width;w++){
        *(array+(idG*height+w))= idG*height+w;
    }
    printf("igG %d    idL %d\n",idG,idL);
}
