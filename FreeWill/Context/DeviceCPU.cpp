#include "Device.h"
#include "../Model/Model.h"

void FreeWill::Device<FreeWill::DeviceType::CPU_NAIVE>::pushWork(FreeWill::WorkerMessage *message)
{
    m_commandQueue.push(message);

    message->thread_id = 1;

}

void FreeWill::Device<FreeWill::DeviceType::CPU_NAIVE>::terminate()
{
    FreeWill::WorkerMessage message(FreeWill::WorkerMessage::Type::TERMINATE);
    pushWork(&message);
    message.join();
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
        FreeWill::WorkerMessage *message = m_commandQueue.pop();

        if (message->workType() == FreeWill::WorkerMessage::Type::TERMINATE)
        {
            message->done();
            break;
        }

        {
            std::unique_lock<std::mutex> ol(outputLock);
            std::cout << "thread: " << this_id << " output." << message->debug_num << std::endl;
        }

        message->done();

    }

    //std::cout << " terminated"<<std::endl;
}

void FreeWill::Device<FreeWill::DeviceType::CPU_NAIVE>::init()
{
    m_workerThread = new std::thread([=]{FreeWill::Device<FreeWill::DeviceType::CPU_NAIVE>::threadLoop();});
}
