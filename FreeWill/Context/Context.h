#ifndef CONTEXT_H
#define CONTEXT_H

#include "../DeviceSelection.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <cstdlib>
#include <type_traits>

namespace FreeWill
{
    template<DeviceType DeviceUsed = CPU_NAIVE>
    class Context
    {
    private:
        Context()
            :m_cudnnHandle(nullptr),
            m_cublasHandle(nullptr),
            m_sharedOneVectorFloat(nullptr),
            m_sharedOneVectorFloatSize(0),
            m_sharedOneVectorDouble(nullptr),
            m_sharedOneVectorDoubleSize(0)
        {}

        cudnnHandle_t m_cudnnHandle;
        cublasHandle_t m_cublasHandle;

        float *m_sharedOneVectorFloat;
        unsigned int m_sharedOneVectorFloatSize;
        double *m_sharedOneVectorDouble;
        unsigned int m_sharedOneVectorDoubleSize;

    public:

        void open()
        {
            int deviceCount;
            cudaGetDeviceCount(&deviceCount);
            int device;
            for (device = 0; device < deviceCount; ++device)
            {
                cudaDeviceProp deviceProp;
                cudaGetDeviceProperties(&deviceProp, device);
                printf("Device %d has compute capability %d.%d.\n", device, deviceProp.major, deviceProp.minor);
                printf("Maximum threads per block: %d\n", deviceProp.maxThreadsPerBlock);
                printf("Device texture alignment: %lu\n", deviceProp.textureAlignment);
                printf("Device texture dimension: %d X %d\n", deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1]);
            }

            size_t freeMem = 0;
            size_t totalMem = 0;
            cudaMemGetInfo(&freeMem, &totalMem);
            printf("available video memory: %ld, %ld (bytes)\n", freeMem, totalMem);

            RUN_CUDNN( cudnnCreate(&m_cudnnHandle));
            RUN_CUBLAS( cublasCreate(&m_cublasHandle));
        }

        void close()
        {
            RUN_CUDNN( cudnnDestroy(m_cudnnHandle));
            RUN_CUBLAS( cublasDestroy(m_cublasHandle));
            cudaDeviceReset();           
        }

        static Context &getSingleton()
        {
            static Context obj;
            return obj;
        }

        const cudnnHandle_t & cudnnHandle() const
        {
            return m_cudnnHandle;
        }

        const cublasHandle_t & cublasHandle() const
        {
            return m_cublasHandle;
        }

        template<typename DataType = float>
        DataType *getSharedOneVector(const unsigned int requestedVectorSize)
        {
           unsigned int *vectorSize = nullptr;
           DataType **vectorBuffer = nullptr;

           if constexpr (std::is_same<DataType, float>::value)
           {
                vectorSize = &m_sharedOneVectorFloatSize;
                vectorBuffer = &m_sharedOneVectorFloat;
           }
           else if constexpr (std::is_same<DataType, double>::value)
           {
                vectorSize = &m_sharedOneVectorDoubleSize;
                vectorBuffer = &m_sharedOneVectorDouble;
           }

           if ((*vectorSize) < requestedVectorSize)
           {
               if (*vectorSize)
               {
                    RUN_CUDA(cudaFree(*vectorBuffer));
               }
               *vectorSize = requestedVectorSize;
               RUN_CUDA(cudaMalloc(vectorBuffer, sizeof(DataType)*(*vectorSize)));
               DataType *initializeBuffer = new DataType[*vectorSize];

               for (unsigned int i =0;i<*vectorSize;++i)
               {
                    initializeBuffer[i] = 1.0;
               }

               RUN_CUDA(cudaMemcpy(*vectorBuffer, initializeBuffer, sizeof(DataType) * (*vectorSize), cudaMemcpyHostToDevice));
               delete [] initializeBuffer;
           }

            return *vectorBuffer;

        }
    };
}
#endif
