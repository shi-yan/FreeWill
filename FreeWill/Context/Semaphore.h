#ifndef SEMAPHORE_H
#define SEMAPHORE_H

#include <mutex>
#include <thread>
#include <condition_variable>

namespace FreeWill
{
    class Semaphore
    {
    private:
        std::mutex m_mutex;
        std::condition_variable m_condition;
        unsigned long m_count = 0; // Initialized as locked.

    public:
        Semaphore(unsigned long initialCount = 0);

        void signal();

        void wait();

        bool tryWait();
    };
}

#endif
