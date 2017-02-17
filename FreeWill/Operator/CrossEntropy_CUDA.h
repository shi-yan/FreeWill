#ifndef CROSSENTROPY_CUDA_H
#define CROSSENTROPY_CUDA_H

template <typename DataType = float>
__host__ void crossEntropyCUDAKernel(DataType *input, DataType *label, DataType *cost, unsigned int labelSize, unsigned int batchSize);



#endif
