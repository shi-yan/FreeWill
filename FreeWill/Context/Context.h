#ifndef CONTEXT_H
#define CONTEXT_H

#include "../DeviceSelection.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <cstdlib>
#include <type_traits>
#include <cstdio>
#include "../Tensor/ReferenceCountedBlob.h"
#include <thread>
#include "Device.h"
#include <iostream>
#include <vector>

namespace FreeWill
{
    template<DeviceType DeviceUsed = DeviceType::CPU_NAIVE>
    class Context
    {
    private:
        Context()
            :m_sharedOneVectorFloat(nullptr),
            m_sharedOneVectorFloatSize(0),
            m_sharedOneVectorDouble(nullptr),
            m_sharedOneVectorDoubleSize(0),
            m_deviceCount(0)
        {}


        float *m_sharedOneVectorFloat;
        unsigned int m_sharedOneVectorFloatSize;
        double *m_sharedOneVectorDouble;
        unsigned int m_sharedOneVectorDoubleSize;
        int m_deviceCount;
        std::vector<Device<DeviceUsed>*> m_deviceList;


    public:

        void open(unsigned int deviceCountOverride = 0)
        {
            if constexpr (DeviceUsed == DeviceType::GPU_CUDA)
            {
                cudaGetDeviceCount(&m_deviceCount);
                for (unsigned int i = 0; i < m_deviceCount; ++i)
                {
                    cudaDeviceProp deviceProp;
                    cudaGetDeviceProperties(&deviceProp, i);
                    printf("Device %d has compute capability %d.%d.\n", i, deviceProp.major, deviceProp.minor);
                    printf("Maximum threads per block: %d\n", deviceProp.maxThreadsPerBlock);
                    printf("Device texture alignment: %lu\n", deviceProp.textureAlignment);
                    printf("Device texture dimension: %d X %d\n", deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1]);

                    Device<DeviceUsed> *device = new Device<DeviceUsed>(i);
                    m_deviceList.push_back(device);

                    device->init();
                }

                size_t freeMem = 0;
                size_t totalMem = 0;
                cudaMemGetInfo(&freeMem, &totalMem);
                printf("available video memory: %ld, %ld (bytes)\n", freeMem, totalMem);

            }
            else if constexpr (DeviceUsed == DeviceType::CPU_NAIVE)
            {
                if (deviceCountOverride == 0)
                {
                    m_deviceCount = std::thread::hardware_concurrency();
                }
                else
                {
                    m_deviceCount = deviceCountOverride;
                }

                std::cout << "CPU count:" << m_deviceCount << std::endl;

                for(int i = 0; i<m_deviceCount; ++i)
                {
                    Device<DeviceUsed> *device = new Device<DeviceUsed>(i);
                    m_deviceList.push_back(device);
                    device->init();

                }



                /*for(int i = 0; i<500;++i)
                {
                    WorkerMessage *messages[100] = {nullptr};

                    for (int e = 0; e<100;++e)
                    {
                        messages[e] = new WorkerMessage(FreeWill::WorkerMessage::Type::NO_WORK);
                        messages[e]->debug_num = i;
                    }


                    for (int e = 0; e<100;++e)
                    {
                        int d = rand() % m_deviceList.size();
                        m_deviceList[d]->pushWork(messages[e]);
                    }


                    for (int e = 0;e<100;++e)
                    {
                        messages[e]->join();
                        delete messages[e];
                        messages[e] = nullptr;
                    }

                    std::cout << "=============================== finished batch: " << i <<std::endl;

                }*/
            }
        }

        void pushWork(unsigned int deviceId, WorkerMessage *message)
        {
            if constexpr (DeviceUsed == DeviceType::CPU_NAIVE)
            {
                m_deviceList[deviceId]->pushWork(message);
            }
        }

        void close()
        {
            if constexpr (DeviceUsed == DeviceType::GPU_CUDA)
            {

            }
            else if constexpr (DeviceUsed == DeviceType::CPU_NAIVE)
            {

                for(int i = 0; i<m_deviceList.size();++i)
                {
                    m_deviceList[i]->terminate();
                    delete m_deviceList[i];
                }

                m_deviceList.clear();
            }
        }

        static Context &getSingleton()
        {
            static Context obj;
            return obj;
        }

        int deviceCount() const
        {
            return m_deviceCount;
        }

        const cudnnHandle_t & cudnnHandle(unsigned int deviceId) const
        {
            if constexpr (DeviceUsed == FreeWill::DeviceType::GPU_CUDA)
            {
                return m_deviceList[deviceId]->cudnnHandle();
            }

            return (cudnnHandle_t)0;
        }

        const cublasHandle_t & cublasHandle(unsigned int deviceId) const
        {
            if constexpr (DeviceUsed == FreeWill::DeviceType::GPU_CUDA)
            {
                return m_deviceList[deviceId]->cublasHandle();
            }
            return (cublasHandle_t)0;
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
