#ifndef DEVICE_H
#define DEVICE_H

#include "../DeviceSelection.h"
#include "../Tensor/ReferenceCountedBlob.h"
#include <thread>
#include <mutex>
#include <condition_variable>
#include "Ringbuffer.h"
#include "WorkerMessage.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <cudnn.h>
#include <cublas_v2.h>

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

        cudnnHandle_t m_cudnnHandle;
        cublasHandle_t m_cublasHandle;
        void threadLoop();

    public:
        Device(unsigned int deviceId = 0)
            : m_workerThread(nullptr),
              m_finished(false),
              m_commandQueue(100),
              m_deviceId(deviceId),
              m_cudaDeviceId(deviceId),
              m_cudnnHandle(nullptr),
              m_cublasHandle(nullptr)
        {}

        const cudnnHandle_t & cudnnHandle() const
        {
            return m_cudnnHandle;
        }

        const cublasHandle_t & cublasHandle() const
        {
            return m_cublasHandle;
        }


        ~Device()
        {
            if (m_workerThread)
            {
                delete m_workerThread;
            }
            RUN_CUDA(cudaSetDevice(m_cudaDeviceId));
            if (m_cudnnHandle)
            {
                RUN_CUDNN( cudnnDestroy(m_cudnnHandle));
            }
            if (m_cublasHandle)
            {
                RUN_CUBLAS( cublasDestroy(m_cublasHandle));
            }
            cudaDeviceReset();
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
