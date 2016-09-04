#ifndef FULLYCONNECTEDLAYERKERNELGPU_H
#define FULLYCONNECTEDLAYERKERNELGPU_H
#include <cuda.h>
#include <cuda_runtime.h>
#include "Global.h"

void FullyConnectedLayerKernelGPU(float *weights,
                                  unsigned int batchSize,
                                  float *inputs,
                                  unsigned int inputSize,
                                  float *outputs,
                                  unsigned int outputSize,
                                  float (*activation) (const float));


void FullyConnectedLayerKernelGPU(double *weights,
                                  unsigned int batchSize,
                                  double *inputs,
                                  unsigned int inputSize,
                                  double *outputs,
                                  unsigned int outputSize,
                                  double (*activation) (const double));

void* getActivationFuncFloat(Activation activation);
void* getActivationFuncDouble(Activation activation);

#endif // FULLYCONNECTEDLAYERKERNELGPU_H
