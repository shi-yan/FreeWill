#include "ElementwiseAdd_CUDA.h"
#include "../DeviceSelection.h"
#include <cuda_runtime.h>

template <typename DataType>
__global__ void elementwiseAdd(DataType *operandA, DataType *operandB, DataType *result, unsigned int size)
{
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if (id < size)
    {
        result[id] = operandA[id] + operandB[id];
    }
}
    
    
template <typename DataType>
__host__ void elementwiseAddCUDAKernel(DataType *operandA, DataType *operandB, DataType *result, unsigned int size)
{
    int blockSize = 1024;
    int gridSize =  size / blockSize + 1;
//    printf("gridsize:%d,%d",gridSize, blockSize);
    elementwiseAdd<DataType><<<gridSize, blockSize>>>(operandA, operandB, result, size);
    CHECK_CUDA_ERROR
}

template __host__ void elementwiseAddCUDAKernel(float *operandA, float *operandB, float *result, unsigned int size); 
template __host__ void elementwiseAddCUDAKernel(double *operandA, double *operandB, double *result, unsigned int size);
