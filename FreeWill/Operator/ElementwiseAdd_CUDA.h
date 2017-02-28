#ifndef ELEMENTWISEADD_CUDA_H
#define ELEMENTWISEADD_CUDA_H

//#ifdef __cplusplus

template <typename DataType = float>
__host__ void elementwiseAddCUDAKernel(DataType *operandA, DataType *operandB, DataType rate, DataType *result, unsigned int size);

//#endif

#endif
