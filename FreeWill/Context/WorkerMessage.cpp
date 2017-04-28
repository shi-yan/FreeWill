#include "WorkerMessage.h"

FreeWill::WorkerMessage::WorkerMessage(Type workType, Operator<DeviceType::CPU_NAIVE> *operatorBase, Model *model)
    :m_workType(workType),
      m_model(model),
      m_operatorBase(operatorBase),
      m_finished(false)
{}

FreeWill::WorkerMessage::WorkerMessage(Type workType, Operator<DeviceType::GPU_CUDA> *operatorBase, Model *model)
    :m_workType(workType),
      m_model(model),
      m_operatorBase(operatorBase),
      m_finished(false)
{}

FreeWill::WorkerMessage::WorkerMessage(const WorkerMessage &in)
    :m_workType(in.m_workType),
      m_model(in.m_model),
      m_finished(false)
{

}

void FreeWill::WorkerMessage::operator =(const WorkerMessage &in)
{
    m_workType = in.m_workType;
    m_model = in.m_model;
    m_finished = in.m_finished;
}

FreeWill::WorkerMessage::~WorkerMessage(){}

void FreeWill::WorkerMessage::join()
{
    std::unique_lock<std::mutex> workLock(m_conditionFinishedMutex);
    m_conditionFinished.wait(workLock, [=]{return m_finished == true;});
    workLock.unlock();
}

void FreeWill::WorkerMessage::done()
{
    std::unique_lock<std::mutex> workLock(m_conditionFinishedMutex);
    m_finished = true;
    //notify must be inside the workLock, otherwise dead lock
    m_conditionFinished.notify_one();
}

FreeWill::WorkerMessage::Type FreeWill::WorkerMessage::workType() const
{
    return m_workType;
}
