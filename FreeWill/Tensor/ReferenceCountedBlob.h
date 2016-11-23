#ifndef REFERENCECOUNTEDBLOB_H
#define REFERENCECOUNTEDBLOB_H

#include "DeviceSelection.h"
#include <cstring>
#include <algorithm>
#include <random>


namespace FreeWill
{
    class ReferenceCounter
    {
    private:
        unsigned int counter;

    public:
        unsigned int increase()
        {
            return ++ counter;
        }

        unsigned int decrease()
        {
            return -- counter;
        }
    };

    template<DeviceType DeviceUsed, typename DataType = float>
    class ReferenceCountedBlob
    {
    private:
        unsigned int m_size;
        ReferenceCounter *m_referenceCounter;
        DataType *m_dataHandle;

        void cleanup()
        {
            if (m_referenceCounter->decrease() == 0)
            {
                delete m_dataHandle;
                delete m_referenceCounter;
            }
            
            m_referenceCounter = nullptr;
            m_dataHandle = nullptr;
            m_size = 0;
        }

    public:
        ReferenceCountedBlob()
            :m_size(0),
            m_referenceCounter(nullptr),
            m_dataHandle(nullptr)
        {
            m_referenceCounter = new ReferenceCounter();
            m_referenceCounter->increase();
        }

        ReferenceCountedBlob(const ReferenceCountedBlob<DeviceUsed, DataType> &blob)
            :m_size(0),
            m_referenceCounter(nullptr),
            m_dataHandle(nullptr)
        {
            if (blob.m_dataHandle) 
            {
                m_referenceCounter = blob.m_referenceCounter;
                m_size = blob.m_size;
                m_dataHandle = blob.m_dataHandle;
                m_referenceCounter->increase();
            }
            else
            {
                m_referenceCounter = new ReferenceCounter();
                m_referenceCounter->increase();
            }
        }

        bool alloc(unsigned int size)
        {
            if constexpr ((DeviceUsed & (CPU_NAIVE | CPU_SIMD)) != 0)
            {
                m_dataHandle = new DataType[size];
                if (m_dataHandle) 
                {
                    m_size = size;
                    std::memset(m_dataHandle, sizeof(DataType) * size, 0);
                    return true;
                }
                else
                {
                    return false;
                }
            } 
            else if constexpr ((DeviceUsed & GPU) != 0) 
            {

            }
        }

        void randomize()
        {
            if constexpr ((DeviceUsed & (CPU_NAIVE | CPU_SIMD)) != 0)
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

        ReferenceCountedBlob<DeviceUsed, DataType> deepCopy() const 
        {
            ReferenceCountedBlob<DeviceUsed, DataType> copy;
            copy.alloc(m_size);

            if constexpr ((DeviceUsed & (CPU_NAIVE | CPU_SIMD)) != 0)
            {
                std::copy(m_dataHandle, m_dataHandle + m_size, copy.m_dataHandle);
            }
            else if constexpr ((DeviceUsed & GPU) != 0)
            {
            }

            return copy;
        }

        void operator=(const ReferenceCountedBlob<DeviceUsed, DataType> &blob)
        {
            if (blob.m_dataHandle)
            {
                cleanup();
                m_referenceCounter = blob.m_referenceCounter;
                m_referenceCounter->increase();
                m_size = blob.m_size;
                m_dataHandle = blob.m_dataHandle;
                
            }
        }

        void operator==(const ReferenceCountedBlob<DeviceUsed, DataType> &blob) const 
        {
            return m_dataHandle == blob.m_dataHandle;
        }

        DataType operator[](unsigned int index) const
        {
            if (m_dataHandle && index < m_size)
            {
                return *(m_dataHandle + index);
            }
            return 0;
        }

        unsigned int size() const
        {
            return m_size;
        }
        
        ~ReferenceCountedBlob()
        {
            cleanup();
        }
    };
}
#endif
