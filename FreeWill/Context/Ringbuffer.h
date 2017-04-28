#ifndef RINGBUFFER_H
#define RINGBUFFER_H

#include "Semaphore.h"
#include <vector>

namespace FreeWill
{

    template <typename ElementType>
    class Ringbuffer
    {
        std::vector<ElementType*> m_buffer;
        Semaphore m_emptySemaphore;
        Semaphore m_fullSemaphore;

        unsigned int m_head;
        unsigned int m_tail;

    public:
        Ringbuffer(unsigned int defaultSize = 100)
            :m_buffer(defaultSize, nullptr),
              m_emptySemaphore(defaultSize),
              m_fullSemaphore(),
                m_head(0),
                m_tail(0)
        {}

        ~Ringbuffer()
        {
            m_buffer.clear();
        }

        ElementType *pop()
        {
            m_fullSemaphore.wait();

            ElementType *element = m_buffer[m_tail];
            m_tail = (m_tail + 1) % m_buffer.size();

            m_emptySemaphore.signal();

            return element;

        }

        void push(ElementType *element)
        {
            m_emptySemaphore.wait();

            m_buffer[m_head] = element;
            m_head = (m_head + 1) % m_buffer.size();

            m_fullSemaphore.signal();
        }
    };
}

#endif
