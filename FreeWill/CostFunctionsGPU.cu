#include "Global.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "CostFunctionsGPU.h"



template<class ScalarType>
__global__ void crossEntropySigmoidKernel(const ScalarType *outputs, unsigned int outputSize, const ScalarType *labels, ScalarType *cost, unsigned int batchSize, ScalarType *derivatives)
{
    int batchId = threadIdx.x;
    ScalarType c = 0.0;
    for(size_t i = 0; i < outputSize; ++i)
    {
        ScalarType output = outputs[outputSize * batchId + i];
        ScalarType label = labels[outputSize*batchId + i];

        c += label*log(output) + (1.0 - label)*log(1.0 - output);
        derivatives[batchId * outputSize + i] = output - label;
    }
    cost[batchId] = -1.0 * c;
}

template<class ScalarType>
__global__ void meanSquaredRectifierKernel(const ScalarType *outputs, unsigned int outputSize, const ScalarType *labels, ScalarType *cost, unsigned int batchSize, ScalarType *derivatives)
{
    int batchId = threadIdx.x;
    ScalarType c = 0.0;
    ScalarType norm = 1.0 / outputSize;
    for(size_t i = 0; i<outputSize; ++i)
    {
        ScalarType output = outputs[outputSize * batchId + i];
        ScalarType label = labels[outputSize*batchId + i];

        c += (output - label) * (output - label);
        derivatives[batchId * outputSize + i] = (output >= 0) ? 2.0 * (output - label) * norm: 0.0;
    }
    cost[batchId] = norm * c;
}

__host__
void crossEntropySigmoidGPUKernel(const float *outputs, unsigned int outputSize, const float *labels, float *cost, unsigned int batchSize, float *derivatives)
{
    crossEntropySigmoidKernel<float><<<1, batchSize>>>(outputs, outputSize, labels, cost, batchSize, derivatives);
}

__host__
void crossEntropySigmoidGPUKernel(const double *outputs, unsigned int outputSize, const double *labels, double *cost, unsigned int batchSize, double *derivatives)
{
    crossEntropySigmoidKernel<double><<<1, batchSize>>>(outputs, outputSize, labels, cost, batchSize, derivatives);
}

__host__
void meanSquaredRectifierGPUKernel(const float *outputs, unsigned int outputSize, const float *labels, float *cost, unsigned int batchSize, float *derivatives)
{
    meanSquaredRectifierKernel<float><<<1, batchSize>>>(outputs, outputSize, labels, cost, batchSize, derivatives);
}

__host__
void meanSquaredRectifierGPUKernel(const double *outputs, unsigned int outputSize, const double *labels, double *cost, unsigned int batchSize, double *derivatives)
{
    meanSquaredRectifierKernel<double><<<1, batchSize>>>(outputs, outputSize, labels, cost, batchSize, derivatives);
}
