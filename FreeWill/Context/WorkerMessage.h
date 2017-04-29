#ifndef WORKERMESSAGE_H
#define WORKERMESSAGE_H

#include <condition_variable>
#include <thread>
#include <mutex>
#include <variant>
#include "../Tensor/ReferenceCountedBlob.h"

namespace FreeWill
{
    class Model;
    template <DeviceType DeviceUsed>
    class Operator;

    class WorkerMessage
    {
    public:
        enum class Type
        {
            NO_WORK,
            FORWARD,
            BACKWARD,
            UPDATE,
            TERMINATE
        };

    private:
        Model *m_model;
        std::variant<Operator<DeviceType::GPU_CUDA>*, Operator<DeviceType::CPU_NAIVE>*> m_operatorBase;
        std::condition_variable m_conditionFinished;
        std::mutex m_conditionFinishedMutex;
        Type m_workType;
        bool m_finished;

    public:
        int debug_num = 0;
        int thread_id = 0;
        WorkerMessage(Type workType = Type::NO_WORK, Operator<DeviceType::CPU_NAIVE> *operatorBase = nullptr,  Model *model = nullptr);
        WorkerMessage(Type workType = Type::NO_WORK, Operator<DeviceType::GPU_CUDA> *operatorBase = nullptr,  Model *model = nullptr);

        WorkerMessage(const WorkerMessage &in);

        void operator=(const WorkerMessage &in);

        ~WorkerMessage();

        void done();

        void join();

        Type workType() const;

        template<DeviceType DeviceUsed = DeviceType::CPU_NAIVE>
        Operator<DeviceUsed> *operatorBase()
        {
            return std::get<Operator<DeviceUsed> *>(m_operatorBase);
        }
    };
}

#endif
