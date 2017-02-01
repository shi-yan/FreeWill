#ifndef DEVICESELECTION_H
#define DEVICESELECTION_H

#define VERIFY_INIT(a) \
    if (! a) \
    {qDebug() << "Operator check failed:" << #a ; abort();}


namespace FreeWill 
{
	typedef enum 
	{
		CPU_NAIVE      = 0x1,
		CPU            = 0x1,
		CPU_SIMD       = 0x2,
		GPU            = 0x4,
		GPU_CUDA       = 0x4
	} DeviceType;
}

#define DEVICE_SPECIFIC(devices) \
    template<DeviceType T = DeviceUsed, typename Enabled = typename std::enable_if<(T & devices) != 0> >

#define RUN_CUDA(cuda_function) \
    if (cuda_function != cudaSuccess) \
        {printf("CUDA error: %s:%d\n", __FILE__, __LINE__);}

#define RUN_CUDNN(cudnn_function) \
    {cudnnStatus_t result = cudnn_function ; \
    if (result != CUDNN_STATUS_SUCCESS) \
        {printf("CUDNN error: %s:%d Error[%d]\n", __FILE__, __LINE__, result);}}

#define RUN_CUBLAS(cublas_function) \
    {cublasStatus_t result = cublas_function ; \
    if (result != CUBLAS_STATUS_SUCCESS) \
        {printf("CUBLAS error: %s:%d Error[%d]\n", __FILE__, __LINE__, result);}}

#define CHECK_CUDA_ERROR \
    if (cudaGetLastError() != cudaSuccess) \
        {printf("CUDA Launch Kernel error: %s:%d\n", __FILE__, __LINE__);}

#endif

#define FAIL_IF(EXP) \
    do { if (EXP) { \
             printf ("Operator integrity check failed: " #EXP ", %s:%d\n", __FILE__, __LINE__); return false;}} \
    while (0)
