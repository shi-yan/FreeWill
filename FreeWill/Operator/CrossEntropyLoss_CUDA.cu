#include "CrossEntropyLoss_CUDA.h"
#include "../DeviceSelection.h"
#include <cuda_runtime.h>


template <typename DataType>
__global__ void crossEntropyLoss(DataType *input, DataType *label, DataType *cost, unsigned int labelSize, unsigned int batchSize)
{
    //printf("========== launch kernel 1 =========\n");
    int id = blockIdx.x*blockDim.x+threadIdx.x;

    if (id < labelSize * batchSize)
    {

        int idInVector = id % labelSize;
        int batchId = id / labelSize;

        DataType temp = -label[batchId * labelSize + idInVector]*log(input[batchId * labelSize + idInVector])
             - (1.0 - label[batchId * labelSize + idInVector])*log(1.0 - input[batchId * labelSize + idInVector]);

        atomicAdd(cost + batchId, temp);
    }
}

#if __CUDA_ARCH__ < 600
__device__ double atomicAddDouble(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}

template<>
__global__ void crossEntropyLoss<double>(double *input, double *label, double *cost, unsigned int labelSize, unsigned int batchSize)
{

    int id = blockIdx.x*blockDim.x+threadIdx.x;

    if (id < labelSize * batchSize)
    {
        int idInVector = id % labelSize;
        int batchId = id / labelSize;

        double temp = -label[batchId * labelSize + idInVector]*log(input[batchId * labelSize + idInVector])
             - (1.0 - label[batchId * labelSize + idInVector])*log(1.0 - input[batchId * labelSize + idInVector]);

        //printf("id:%d idinvec %d batchid%d\n",id, idInVector, batchId);
        atomicAddDouble(cost + batchId, temp);
    }


}

#endif

    
    
template <typename DataType>
__host__ void crossEntropyLossCUDAKernel(DataType *input, DataType *label, DataType *cost, unsigned int labelSize, unsigned int batchSize)
{
    int blockSize = 1024;
    int gridSize =  (labelSize * batchSize) / blockSize ;

    if ((labelSize * batchSize) % blockSize != 0)
    {
        gridSize += 1;
    }

    cudaMemset(cost, 0, sizeof(DataType) * batchSize);
    //printf("%d, %d",cost, sizeof(DataType) * batchSize);
    CHECK_CUDA_ERROR
    //printf("gridsize:%d blocksize:%d labelsize %d batchsize %d\n",gridSize,blockSize,labelSize,batchSize);
    crossEntropyLoss<DataType><<<gridSize, blockSize>>>(input, label, cost, labelSize, batchSize);
    CHECK_CUDA_ERROR
}

template __host__ void crossEntropyLossCUDAKernel(float *input, float *label, float *cost, unsigned int labelSize, unsigned int batchSize);
//The kernel for double type is disabled, because the function atomicAdd is unavailable when cc < 6.0
//#if __CUDA_ARCH__ >= 600
template __host__ void crossEntropyLossCUDAKernel(double *input, double *label, double *cost, unsigned int labelSize, unsigned int batchSize);
//#endif

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
__host__ void sigmoidCrossEntropyLossDerivativeCUDAKernel(DataType *input, DataType *label, DataType *output, unsigned int size)
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

template __host__ void sigmoidCrossEntropyLossDerivativeCUDAKernel(float *input, float *label, float *output, unsigned int size);
template __host__ void sigmoidCrossEntropyLossDerivativeCUDAKernel(double *input, double *label, double *output, unsigned int size);


