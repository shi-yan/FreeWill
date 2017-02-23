#ifndef CROSSENTROPYLOSS_CUDA_H
#define CROSSENTROPYLOSS_CUDA_H

template <typename DataType = float>
__host__ void crossEntropyLossCUDAKernel(DataType *input, DataType *label, DataType *cost, unsigned int labelSize, unsigned int batchSize);

template <typename DataType = float>
__host__ void sigmoidCrossEntropyLossDerivativeCUDAKernel(DataType *input, DataType *label, DataType *output, unsigned int size);



#endif
