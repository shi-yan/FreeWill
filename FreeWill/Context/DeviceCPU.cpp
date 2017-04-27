#include "Device.h"
#include "../Model/Model.h"

void FreeWill::Device<FreeWill::DeviceType::CPU_NAIVE>::pushWork(WorkType workType, Model *model)
{

}

void FreeWill::Device<FreeWill::DeviceType::CPU_NAIVE>::threadLoop()
{
    /*std::thread::id this_id = std::this_thread::get_id();

    std::cout << "thread" << this_id << "started" << std::endl;

    while(!m_finished)
    {
        std::unique_lock<std::mutex> workLock(m_conditionNewWorkAvailable);
        m_conditionNewWorkAvailable.wait(workLock, []{return m_workType != WorkType::NO_WORK;});
    }*/
}
