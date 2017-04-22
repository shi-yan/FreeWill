#ifndef DEVICESELECTION_H
#define DEVICESELECTION_H

#include <cmath>

#define VERIFY_INIT(a) \
    if (! a) \
    {qDebug() << "Operator check failed:" << #a ; abort();}

#define DEVICE_SPECIFIC(devices) \
    template<DeviceType T = DeviceUsed, typename Enabled = typename std::enable_if<(T & devices) != 0> >

#define RUN_CUDA(cuda_function) \
    if (cuda_function != cudaSuccess) \
        {printf("CUDA error: %s:%d\n", __FILE__, __LINE__);}

#define RUN_CUDNN(cudnn_function) \
    {cudnnStatus_t result = cudnn_function ; \
    if (result != CUDNN_STATUS_SUCCESS) \
        {printf("CUDNN error: %s:%d Error[%d %s]\n", __FILE__, __LINE__, result, cudnnGetErrorString(result));}}

#define RUN_CUBLAS(cublas_function) \
    {cublasStatus_t result = cublas_function ; \
    if (result != CUBLAS_STATUS_SUCCESS) \
        {printf("CUBLAS error: %s:%d Error[%d]\n", __FILE__, __LINE__, result);}}

#define CHECK_CUDA_ERROR \
    {cudaError_t result = cudaGetLastError(); \
    if (result != cudaSuccess) \
        {printf("CUDA Launch Kernel error: %s:%d Error[%s]\n", __FILE__, __LINE__, cudaGetErrorString(result));}}

#define ENUM_CASE(option, message) \
    case option: qDebug() << message << #option << option; break;

static const double epsilon = 1e-4;

static inline double relativeError(double fakeGradient, double realGradient)
{
    double norm = std::fmax(std::abs(fakeGradient), std::abs(realGradient));
    if (norm == 0.0)
    {
        norm = 1.0;
    }
    return std::abs(fakeGradient - realGradient) / norm;
}

#endif
