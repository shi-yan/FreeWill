#ifndef REFERENCECOUNTEDBLOB_H
#define REFERENCECOUNTEDBLOB_H

#include "DeviceSelection.h"
#include <cstring>
#include <algorithm>
#include <random>
#include <cstdio>

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

        int decrease()
        {
            return -- counter;
        }
    };

    template<DeviceType DeviceUsed>
    class ReferenceCountedBlob
    {
    private:
        unsigned int m_sizeInByte;
        ReferenceCounter *m_referenceCounter;
        unsigned char *m_dataHandle;

        void cleanup()
        {
            if (m_referenceCounter->decrease() == 0)
            {
                free(m_dataHandle);
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
            m_dataHandle(nullptr)
        {
            m_referenceCounter = new ReferenceCounter();
            m_referenceCounter->increase();
        }

        ReferenceCountedBlob(const ReferenceCountedBlob<DeviceUsed> &blob)
            :m_sizeInByte(0),
            m_referenceCounter(nullptr),
            m_dataHandle(nullptr)
        {
            if (blob.m_dataHandle) 
            {
                m_referenceCounter = blob.m_referenceCounter;
                m_sizeInByte = blob.m_sizeInByte;
                m_dataHandle = blob.m_dataHandle;
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
            std::memset(m_dataHandle, m_sizeInByte, 0);
        }

        unsigned char * dataHandle() 
        {
            return m_dataHandle;
        }

        bool alloc(unsigned int sizeInByte)
        {
            if constexpr ((DeviceUsed & (CPU_NAIVE | CPU_SIMD)) != 0)
            {
                m_dataHandle = (unsigned char *) malloc(sizeInByte);
                if (m_dataHandle) 
                {
                    m_sizeInByte = sizeInByte;
                    std::memset(m_dataHandle, sizeInByte, 0);
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

/*        void randomize()
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
*/
        ReferenceCountedBlob<DeviceUsed> deepCopy() const 
        {
            ReferenceCountedBlob<DeviceUsed> copy;
            copy.alloc(m_sizeInByte);

            if constexpr ((DeviceUsed & (CPU_NAIVE | CPU_SIMD)) != 0)
            {
                std::copy(m_dataHandle, m_dataHandle + m_sizeInByte, copy.m_dataHandle);
            }
            else if constexpr ((DeviceUsed & GPU) != 0)
            {
            }

            return copy;
        }

        void operator=(const ReferenceCountedBlob<DeviceUsed> &blob)
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

        void operator==(const ReferenceCountedBlob<DeviceUsed> &blob) const 
        {
            return m_dataHandle == blob.m_dataHandle;
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
