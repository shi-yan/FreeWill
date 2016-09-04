#ifndef COSTFUNCTIONSGPU_H
#define COSTFUNCTIONSGPU_H

//void *getCostFunctionByName();

void crossEntropySigmoidGPUKernel(const float *outputs, unsigned int outputSize, const float *labels, float *cost, unsigned int batchSize, float *derivatives);
void crossEntropySigmoidGPUKernel(const double *outputs, unsigned int outputSize, const double *labels, double *cost, unsigned int batchSize, double *derivatives);


void meanSquaredRectifierGPUKernel(const float *outputs, unsigned int outputSize, const float *labels, float *cost, unsigned int batchSize, float *derivatives);
void meanSquaredRectifierGPUKernel(const double *outputs, unsigned int outputSize, const double *labels, double *cost, unsigned int batchSize, double *derivatives);


#endif // COSTFUNCTIONSGPU_H
