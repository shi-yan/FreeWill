#include "CrossEntropy_CUDA.h"
#include "../DeviceSelection.h"
#include <cuda_runtime.h>

template <typename DataType>
__global__ void crossEntropy(DataType *input, DataType *label, DataType *cost, unsigned int labelSize, unsigned int batchSize)
{
    int id = blockIdx.x*blockDim.x+threadIdx.x;

    int idInVector = id % labelSize;
    int batchId = id / labelSize;

    DataType temp = -label[batchId * labelSize + idInVector]*log(input[batchId * labelSize + idInVector])                                                            
         - (1.0 - label[batchId * labelSize + idInVector])*log(1.0 - input[batchId * labelSize + idInVector]);

    atomicAdd(cost + batchId, temp);
}
    
    
template <typename DataType>
__host__ void crossEntropyCUDAKernel(DataType *input, DataType *label, DataType *cost, unsigned int labelSize, unsigned int batchSize)
{
    int blockSize = 1024;
    int gridSize =  (labelSize * batchSize) / blockSize ;

    if ((labelSize * batchSize) % blockSize != 0)
    {
        gridSize += 1;
    }

    cudaMemset(cost, 0, sizeof(DataType) * batchSize);
    crossEntropy<DataType><<<gridSize, blockSize>>>(input, label, cost, labelSize, batchSize);
    CHECK_CUDA_ERROR
}

template __host__ void crossEntropyCUDAKernel(float *input, float *label, float *cost, unsigned int labelSize, unsigned int batchSize); 
//The kernel for double type is disabled, because the function atomicAdd is unavailable when cc < 6.0
#if __CUDA_ARCH__ >= 600
template __host__ void crossEntropyCUDAKernel(double *input, double *label, double *cost, unsigned int labelSize, unsigned int batchSize);
#endif

template <typename DataType>
__global__ void elementwiseSub(DataType *operandA, DataType *operandB, DataType *result, unsigned int size)
{
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if (id < size)
    {
        result[id] = operandA[id] - operandB[id];
    }
}
        
template <typename DataType>
__host__ void sigmoidCrossEntropyDerivativeCUDAKernel(DataType *input, DataType *label, DataType *output, unsigned int size)
{
    int blockSize = 1024;
    int gridSize =  size / blockSize ;

    if (size % blockSize != 0)
    {
        gridSize += 1;
    }

    elementwiseSub<DataType><<<gridSize, blockSize>>>(input, label, output, size);
    CHECK_CUDA_ERROR
}

template __host__ void sigmoidCrossEntropyDerivativeCUDAKernel(float *input, float *label, float *output, unsigned int size);
template __host__ void sigmoidCrossEntropyDerivativeCUDAKernel(double *input, double *label, double *output, unsigned int size);


