#ifndef SOFTMAXLOSS_CUDA_H
#define SOFTMAXLOSS_CUDA_H

//#ifdef __cplusplus

template <typename DataType = float>
__host__ void softmaxLossCUDAKernel(DataType *output, unsigned int *label, DataType *cost, unsigned int vectorSize, unsigned int batchSize);

//#endif

#endif
