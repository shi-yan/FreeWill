#ifndef WORKERMESSAGE_H
#define WORKERMESSAGE_H

#include <condition_variable>
#include <thread>
#include <mutex>

namespace FreeWill
{
    class Model;
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
        std::condition_variable m_conditionFinished;
        std::mutex m_conditionFinishedMutex;
        Type m_workType;
        bool m_finished;

    public:
        int debug_num = 0;
        int thread_id = 0;
        WorkerMessage(Type workType = Type::NO_WORK, Model *model = nullptr);
        WorkerMessage(const WorkerMessage &in);

        void operator=(const WorkerMessage &in);

        ~WorkerMessage();

        void done();

        void join();

        Type workType() const;
    };
}

#endif
