
#include "Global.h"
#include "ActivationFunctions.h"
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32

template<typename ScalarType>
__global__ void FullyConnectedLayerKernel(ScalarType *weights, unsigned int batchSize, ScalarType *inputs, unsigned int inputSize, ScalarType *outputs, unsigned int outputSize, ScalarType (*activation) (const ScalarType), unsigned int numblock)
{
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    int row = threadIdx.y;
    int col = threadIdx.x;
    unsigned int rx = blockCol * BLOCK_SIZE + col;
    unsigned int ry = row + blockRow * BLOCK_SIZE;

    float Cvalue = 0;

    for (int m = 0; m < numblock ; ++m)
    {
        __shared__ ScalarType As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ ScalarType Bs[BLOCK_SIZE][BLOCK_SIZE];

        unsigned int x = m*BLOCK_SIZE;
        unsigned int y = row + x;
        x += col;

        As[row][col] = (ry < batchSize && x< inputSize) ? inputs[ry * inputSize + x] : 0;
        Bs[row][col] = (rx < outputSize && y < inputSize) ? weights[y * outputSize + rx] : 0;

        __syncthreads();

        for (int e = 0; e < BLOCK_SIZE; ++e)
        {
            Cvalue += As[row][e] * Bs[e][col];
        }

        __syncthreads();
    }


    Cvalue = (rx< outputSize &&  ry<batchSize)? ( outputs[ry * outputSize + rx] = activation(Cvalue)):0;
}


__host__
void FullyConnectedLayerKernelGPU(float *weights,
                                  unsigned int batchSize,
                                  float *inputs,
                                  unsigned int inputSize,
                                  float *outputs,
                                  unsigned int outputSize,
                                  float (*activation) (const float))
{
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((outputSize / dimBlock.x) + ((outputSize%dimBlock.x) == 0? 0 : 1), (batchSize / dimBlock.y) + ((batchSize%dimBlock.y) == 0? 0:1));
    unsigned int numblock = (inputSize / BLOCK_SIZE) + (((inputSize % BLOCK_SIZE) == 0) ? 0 : 1);

    FullyConnectedLayerKernel<float><<<dimGrid, dimBlock>>>(weights,
                                                                 batchSize,
                                                                 inputs,
                                                                 inputSize,
                                                                 outputs,
                                                                 outputSize,
                                                                 activation,
                                                            numblock);


}

__host__
void FullyConnectedLayerKernelGPU(double *weights,
                                  unsigned int batchSize,
                                  double *inputs,
                                  unsigned int inputSize,
                                  double *outputs,
                                  unsigned int outputSize,
                                  double (*activation) (const double))
{
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((outputSize / dimBlock.x) + ((outputSize%dimBlock.x) == 0? 0 : 1), (batchSize / dimBlock.y) + ((batchSize%dimBlock.y) == 0? 0:1));
    unsigned int numblock = (inputSize / BLOCK_SIZE) + (((inputSize % BLOCK_SIZE) == 0) ? 0 : 1);

    FullyConnectedLayerKernel<double><<<dimGrid, dimBlock>>>(weights,
                                                                 batchSize,
                                                                 inputs,
                                                                 inputSize,
                                                                 outputs,
                                                                 outputSize,
                                                                 activation,
                                                            numblock);


}

__host__
void* getActivationFuncFloat(Activation activation)
{
    void *m_activation = 0;

    switch(activation)
    {
    case Sigmoid:
        cudaMemcpyFromSymbol(&m_activation, fptr_sigmoid, sizeof(void *));
        break;
    case Rectifier:
        cudaMemcpyFromSymbol(&m_activation, fptr_rectifier, sizeof(void *));
        break;
    case Tanh:
        cudaMemcpyFromSymbol(&m_activation, fptr_tanh, sizeof(void *));
        break;
    case None:
    default:
        cudaMemcpyFromSymbol(&m_activation, fptr_noActivation, sizeof(void *));
    }

    return m_activation;
}

__host__
void* getActivationFuncDouble(Activation activation)
{
    void *m_activation = 0;

    switch(activation)
    {
    case Sigmoid:
        cudaMemcpyFromSymbol(&m_activation, fptr_sigmoid_d, sizeof(void *));
        break;
    case Rectifier:
        cudaMemcpyFromSymbol(&m_activation, fptr_rectifier_d, sizeof(void *));
        break;
    case Tanh:
        cudaMemcpyFromSymbol(&m_activation, fptr_tanh_d, sizeof(void *));
        break;
    case None:
    default:
        cudaMemcpyFromSymbol(&m_activation, fptr_noActivation_d, sizeof(void *));
    }

    return m_activation;
}
