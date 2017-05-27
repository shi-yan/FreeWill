#ifndef DEVICE_H
#define DEVICE_H

#include "../DeviceSelection.h"
#include "../Tensor/ReferenceCountedBlob.h"
#include <thread>
#include <mutex>
#include <condition_variable>
#include "Ringbuffer.h"
#include "WorkerMessage.h"

namespace FreeWill
{



    class Model;

    template<DeviceType DeviceUsed>
    class Device
    {
    public:
        int m_debugId;
    };

    template<>
    class Device<DeviceType::GPU_CUDA>
    {
    private:
        std::thread *m_workerThread;
        bool m_finished = false;

        Ringbuffer<WorkerMessage> m_commandQueue;
        unsigned int m_deviceId;
        unsigned int m_cudaDeviceId;

        void threadLoop();

    public:
        Device(unsigned int deviceId = 0)
            : m_workerThread(nullptr),
              m_finished(false),
              m_commandQueue(100),
              m_deviceId(deviceId),
              m_cudaDeviceId(deviceId)
        {}

        ~Device()
        {
            if (m_workerThread)
            {
                delete m_workerThread;
            }
        }

        void pushWork(WorkerMessage *message);

        void init();

        void terminate();

    };

    template<>
    class Device<DeviceType::CPU_NAIVE>
    {

    private:
        std::thread *m_workerThread;
        bool m_finished = false;
        Ringbuffer<WorkerMessage> m_commandQueue;
        unsigned int m_deviceId;

        void threadLoop();

    public:
        Device(unsigned int deviceId = 0)
            : m_workerThread(nullptr),
              m_finished(false),
              m_commandQueue(100),
              m_deviceId(deviceId)
        {
        }

        ~Device()
        {
            if (m_workerThread)
            {
                delete m_workerThread;
            }
        }

        void pushWork(WorkerMessage *message);

        void init();

        void terminate();
    };
}


#endif
