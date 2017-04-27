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
        UPDATE,
        TERMINATE
    };

    class Model;

    class WorkerMessage
    {
    private:
        Model *m_model;
        std::condition_variable m_conditionFinished;
        std::mutex m_busyLock;
        WorkType m_workType;

    }

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
        std::thread *m_workerThread;
        bool m_finished = false;
        Model *m_model;
        WorkType m_workType;
        std::mutex m_busyLock;
        std::condition_variable m_conditionNewWorkAvailable;

        std::vector<WorkerMessage> m_messagePool;


        void threadLoop();

    public:
        Device()
            : m_workerThread(nullptr),
              m_finished(false),
              m_model(nullptr),
              m_workType(WorkType::NO_WORK)
        {
        }

        ~Device()
        {
            if (m_workerThread)
            {
                delete m_workerThread;
            }
        }

        void pushWork(WorkType workType, Model *model);

        void init();

        void terminate();

    };
}


#endif
