#ifndef DEVICE_H
#define DEVICE_H

#include "../DeviceSelection.h"
#include "../Tensor/ReferenceCountedBlob.h"
#include <thread>
#include <mutex>

namespace FreeWill
{

    enum class WorkType
    {
        NO_WORK,
        FORWARD,
        BACKWARD,
        UPDATE
    };

    class Model;

    template<DeviceType DeviceUsed>
    class Device{};

    template<>
    class Device<DeviceType::GPU_CUDA>
    {

    };

    template<>
    class Device<DeviceType::CPU_NAIVE>
    {

    private:
        std::thread m_workerThread;
        bool m_finished = false;
        Model *m_model;
        WorkType m_workType;
        std::mutex m_busyLock;
        std::mutex m_conditionNewWorkAvailable;


        void threadLoop();

    public:
        Device()
            : m_workerThread(this->threadLoop),
              m_finished(false),
              m_model(nullptr),
              m_workType(WorkType::NO_WORK)
        {
        }

        ~Device(){}

        void pushWork(WorkType workType, Model *model);

    };
}


#endif
