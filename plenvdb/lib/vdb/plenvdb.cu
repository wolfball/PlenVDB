#include "plenvdb.cuh"

void gpuAssert(cudaError_t code, const char *file, int line, bool abort=false)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if( abort )
            exit(code);
    }
}

void gpuCheckKernelExecutionError( const char *file, int line)
{
    gpuAssert( cudaPeekAtLastError(), file, line);
    gpuAssert( cudaDeviceSynchronize(), file, line);    
}