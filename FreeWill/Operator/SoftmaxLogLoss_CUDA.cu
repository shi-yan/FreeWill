#include "SoftmaxLogLoss_CUDA.h"
#include "../DeviceSelection.h"
#include <cuda_runtime.h>

template <typename DataType>
__global__ void softmaxLogLoss(DataType *output, unsigned int *label, DataType *cost, unsigned int vectorSize, unsigned int batchSize)
{
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    int size = vectorSize * batchSize;
    if (id < size)
    {
        int vectorId = id % vectorSize;
        int batchId = id / vectorSize;

        if (label[batchId] == vectorId)
        {
            cost[batchId] = -log(output[vectorId]);
        }
    }
}


template <typename DataType>
__host__ void softmaxLogLossCUDAKernel(DataType *output, unsigned int *label, DataType *cost, unsigned int vectorSize, unsigned int batchSize)
{
    int blockSize = 1024;
    int size = vectorSize * batchSize;
    int gridSize =  size / blockSize ;

    if (size % blockSize != 0)
    {
        gridSize += 1;
    }

//    printf("gridsize:%d,%d",gridSize, blockSize);
    softmaxLogLoss<DataType><<<gridSize, blockSize>>>(output, label, cost, vectorSize, batchSize);
    CHECK_CUDA_ERROR
}

template __host__ void softmaxLogLossCUDAKernel(float *output, unsigned int *label, float *cost, unsigned int vectorSize, unsigned int batchSize);
template __host__ void softmaxLogLossCUDAKernel(double *output, unsigned int *label, double *cost, unsigned int vectorSize, unsigned int batchSize);


template <typename DataType>
__global__ void softmaxLogLossDerivative(DataType *inputDelta, DataType *output, unsigned int *label, unsigned int vectorSize, unsigned int batchSize)
{
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    int size = vectorSize * batchSize;
    if (id < size)
    {
        int vectorId = id % vectorSize;
        int batchId = id / vectorSize;

        inputDelta[batchId*vectorSize+ vectorId] = output[batchId*vectorSize +vectorId] + ((label[batchId] == vectorId)?-1.0:0.0);
    }
}

template <typename DataType>
__host__ void softmaxLogLossDerivativeCUDAKernel(DataType *inputDelta, DataType *output, unsigned int *label, unsigned int vectorSize, unsigned int batchSize)
{
    int blockSize = 1024;
    int size = vectorSize * batchSize;
    int gridSize =  size / blockSize ;

    if (size % blockSize != 0)
    {
        gridSize += 1;
    }

    softmaxLogLossDerivative<DataType><<<gridSize, blockSize>>>(inputDelta, output, label, vectorSize, batchSize);
    CHECK_CUDA_ERROR
}

template __host__ void softmaxLogLossDerivativeCUDAKernel(float *inputDelta, float *output, unsigned int *label, unsigned int vectorSize, unsigned int batchSize);
template __host__ void softmaxLogLossDerivativeCUDAKernel(double *inputDelta, double *output, unsigned int *label, unsigned int vectorSize, unsigned int batchSize);
