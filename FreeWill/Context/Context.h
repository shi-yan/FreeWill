#ifndef CONTEXT_H
#define CONTEXT_H

#include "../DeviceSelection.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <cudnn.h>

namespace FreeWill
{
    template<DeviceType DeviceUsed = CPU_NAIVE>
    class Context
    {
    private:
        Context(){}

        cudnnHandle_t m_cudnnHandle;


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
        }

        void close()
        {
            RUN_CUDNN( cudnnDestroy(m_cudnnHandle));
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
    };
}
#endif
