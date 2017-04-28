#include "Semaphore.h"

void FreeWill::Semaphore::signal()
{
    std::unique_lock<std::mutex> lock(m_mutex);
    ++m_count;
    m_condition.notify_one();
}

void FreeWill::Semaphore::wait()
{
    std::unique_lock<std::mutex> lock(m_mutex);
    m_condition.wait(lock,  [=]{return (m_count > 0);});
    --m_count;
}

bool FreeWill::Semaphore::tryWait()
{
    std::unique_lock<std::mutex> lock(m_mutex);
    if(m_count)
    {
        --m_count;
        return true;
    }
    return false;
}

FreeWill::Semaphore::Semaphore(unsigned long initialCount)
    :m_count(initialCount)
{}
