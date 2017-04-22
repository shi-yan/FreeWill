#ifndef REFERENCECOUNTEDBLOB_H
#define REFERENCECOUNTEDBLOB_H

#include "DeviceSelection.h"
#include <cstring>
#include <algorithm>
#include <random>
#include <cstdio>

#include <cuda_runtime.h>
#include <cstdint>



namespace FreeWill
{
    enum class DeviceType : uint32_t
    {
        CPU_NAIVE      = 0x1,
        GPU_CUDA       = 0x4
    };

    template <DeviceType DeviceUsed>
    class TensorBase;

    class ReferenceCounter
    {
    private:
        unsigned int counter;

    public:
        unsigned int increase()
        {
            return ++ counter;
        }

        int decrease()
        {
            return -- counter;
        }
    };

    template<DeviceType DeviceUsed>
    class ReferenceCountedBlob
    {
    private:
        friend class TensorBase<DeviceUsed>;
        unsigned int m_sizeInByte;
        ReferenceCounter *m_referenceCounter;
        unsigned char *m_dataHandle;

        void *m_gpuDataHandle;

        void cleanup()
        {
            if (m_referenceCounter->decrease() == 0)
            {
                free(m_dataHandle);
                if constexpr (DeviceUsed == DeviceType::GPU_CUDA)
                {
                    if (m_gpuDataHandle)
                    {
                        RUN_CUDA(cudaFree(m_gpuDataHandle))
                    }
                }
                delete m_referenceCounter;
            }
           
            m_referenceCounter = nullptr;
            m_dataHandle = nullptr;
            m_sizeInByte = 0;
        }

    public:
        ReferenceCountedBlob()
            :m_sizeInByte(0),
            m_referenceCounter(nullptr),
            m_dataHandle(nullptr),
            m_gpuDataHandle(nullptr)
        {
            m_referenceCounter = new ReferenceCounter();
            m_referenceCounter->increase();
        }

        ReferenceCountedBlob(const ReferenceCountedBlob<DeviceUsed> &blob)
            :m_sizeInByte(0),
            m_referenceCounter(nullptr),
            m_dataHandle(nullptr),
            m_gpuDataHandle(nullptr)
        {
            if (blob.m_dataHandle) 
            {
                m_referenceCounter = blob.m_referenceCounter;
                m_sizeInByte = blob.m_sizeInByte;
                m_dataHandle = blob.m_dataHandle;
                m_gpuDataHandle = blob.m_gpuDataHandle;
                m_referenceCounter->increase();
            }
            else
            {
                m_referenceCounter = new ReferenceCounter();
                m_referenceCounter->increase();
            }
        }

        void clear()
        {
            std::memset(m_dataHandle, 0, m_sizeInByte);

            if constexpr (DeviceUsed == DeviceType::GPU_CUDA)
            {
                RUN_CUDA(cudaMemset(m_gpuDataHandle, 0, m_sizeInByte));
            }
        }

        unsigned char * dataHandle() 
        {
            return m_dataHandle;
        }

        unsigned char * gpuDataHandle()
        {
            return m_gpuDataHandle;
        }

        bool alloc(unsigned int sizeInByte)
        {
            if constexpr (DeviceUsed == DeviceType::CPU_NAIVE)
            {
                m_dataHandle = (unsigned char *) malloc(sizeInByte);
                if (m_dataHandle) 
                {
                    m_sizeInByte = sizeInByte;

                    //printf("memset: %d\n", sizeInByte);
                    std::memset(m_dataHandle,0,  sizeInByte);

                    //printf("memset result: %d, %d, %d, %d, %d\n", m_dataHandle[0], m_dataHandle[1], m_dataHandle[2], m_dataHandle[3], m_dataHandle[4]);
                    return true;
                }
                else
                {
                    return false;
                }
            } 
            else if constexpr (DeviceUsed == DeviceType::GPU_CUDA)
            {
                RUN_CUDA(cudaMalloc(&m_gpuDataHandle, sizeInByte));
                m_dataHandle = (unsigned char *) malloc(sizeInByte);
                if (m_gpuDataHandle && m_dataHandle)
                {
                    m_sizeInByte = sizeInByte;
                    
                    RUN_CUDA(cudaMemset(m_gpuDataHandle, 0, sizeInByte));
                    std::memset(m_dataHandle, 0, sizeInByte);
                    return true;
                }
                else
                {
                    if (m_gpuDataHandle)
                    {
                        RUN_CUDA(cudaFree(m_gpuDataHandle));
                        m_gpuDataHandle = nullptr;
                    }
                    if (m_dataHandle)
                    {
                        std::free(m_dataHandle);
                        m_dataHandle = nullptr;
                    }
                    return false;
                }
            }
        }

/*        void randomize()
        {
            if constexpr ((DeviceUsed & (CPU_NAIVE)) != 0)
            {
                 std::random_device rd;
                 std::mt19937 gen(rd());
                 std::uniform_real_distribution<> dis(0, 1);
                 for (int n = 0; n < m_size; ++n) 
                 {
                     m_dataHandle[n] = dis(gen);
                 }
            }
        }
*/
        ReferenceCountedBlob<DeviceUsed> deepCopy() const 
        {
            ReferenceCountedBlob<DeviceUsed> copy;
            copy.alloc(m_sizeInByte);

            if constexpr (DeviceUsed == DeviceType::CPU_NAIVE)
            {
                std::copy(m_dataHandle, m_dataHandle + m_sizeInByte, copy.m_dataHandle);
            }
            else if constexpr (DeviceUsed == DeviceType::GPU_CUDA)
            {
                std::copy(m_dataHandle, m_dataHandle + m_sizeInByte, copy.m_dataHandle);
                RUN_CUDA(cudaMemcpy(copy.m_gpuDataHandle, m_gpuDataHandle, m_sizeInByte, cudaMemcpyDeviceToDevice));
            }

            return copy;
        }

        void operator=(const ReferenceCountedBlob<DeviceUsed> &blob)
        {
            if constexpr (DeviceUsed == DeviceType::CPU_NAIVE)
            {
                if (blob.m_dataHandle)
                {
                    cleanup();
                    m_referenceCounter = blob.m_referenceCounter;
                    m_referenceCounter->increase();
                    m_sizeInByte = blob.m_sizeInByte;
                    m_dataHandle = blob.m_dataHandle;
                }
            }
            else if constexpr (DeviceUsed == DeviceType::GPU_CUDA)
            {
                if (blob.m_gpuDataHandle)
                {
                    cleanup();
                    m_referenceCounter = blob.m_referenceCounter;
                    m_referenceCounter->increase();
                    m_sizeInByte = blob.m_sizeInByte;
                    m_dataHandle = blob.m_dataHandle;
                    m_gpuDataHandle = blob.m_gpuDataHandle;
                }
            }
        }

        bool operator==(const ReferenceCountedBlob<DeviceUsed> &blob) const 
        {
            if constexpr (DeviceUsed == DeviceType::CPU_NAIVE)
            {
                return m_dataHandle == blob.m_dataHandle;
            }
            else if constexpr (DeviceUsed == DeviceType::GPU_CUDA)
            {
                return m_gpuDataHandle == blob.m_gpuDataHandle;
            }
        }

        void copyFromHostToDevice()
        {
            if constexpr (DeviceUsed == DeviceType::GPU_CUDA)
            {
                RUN_CUDA(cudaMemcpy(m_gpuDataHandle, m_dataHandle, m_sizeInByte, cudaMemcpyHostToDevice));
            }
        }

        void copyFromDeviceToHost()
        {
            if constexpr (DeviceUsed == DeviceType::GPU_CUDA)
            {
                RUN_CUDA(cudaMemcpy(m_dataHandle, m_gpuDataHandle, m_sizeInByte, cudaMemcpyDeviceToHost));
            }
        }

        unsigned char operator[](unsigned int index) const
        {
            if (m_dataHandle && index < m_sizeInByte)
            {
                return *(m_dataHandle + index);
            }
            return 0;
        }

        unsigned int sizeInByte() const
        {
            return m_sizeInByte;
        }
        
        ~ReferenceCountedBlob()
        {
            cleanup();
        }
    };
}
#endif
