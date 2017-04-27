#include "Device.h"
#include "../Model/Model.h"

void FreeWill::Device<FreeWill::DeviceType::CPU_NAIVE>::pushWork(WorkType workType, Model *model)
{
    std::unique_lock<std::mutex> workLock(m_busyLock);
    m_conditionNewWorkAvailable.wait(workLock, [=]{return m_workType == WorkType::NO_WORK;});
    m_workType = workType;
    workLock.unlock();
    m_conditionNewWorkAvailable.notify_one();
}

void FreeWill::Device<FreeWill::DeviceType::CPU_NAIVE>::terminate()
{
    pushWork(WorkType::TERMINATE, nullptr);
    m_workerThread->join();
    delete m_workerThread;
    m_workerThread = nullptr;
}

std::mutex outputLock;

void FreeWill::Device<FreeWill::DeviceType::CPU_NAIVE>::threadLoop()
{
    std::thread::id this_id = std::this_thread::get_id();

    while(!m_finished)
    {

        std::unique_lock<std::mutex> workLock(m_busyLock);
        m_conditionNewWorkAvailable.wait(workLock, [=]{return m_workType != WorkType::NO_WORK;});

        if (m_workType == WorkType::TERMINATE)
        {
            workLock.unlock();
            break;
        }

        {
            std::unique_lock<std::mutex> ol(outputLock);
            std::cout << "thread: " << this_id << "output" << std::endl;
        }

        m_workType = WorkType::NO_WORK;
        workLock.unlock();
        m_conditionNewWorkAvailable.notify_one();

    }

    //std::cout <<
}

void FreeWill::Device<FreeWill::DeviceType::CPU_NAIVE>::init()
{
    m_workerThread = new std::thread([=]{FreeWill::Device<FreeWill::DeviceType::CPU_NAIVE>::threadLoop();});
    //m_workerThread->join();
    //delete thread;
}
