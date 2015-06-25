#ifndef WORD2VECMODELS
#define WORD2VECMODELS

#include <string>
#include <functional>
#include <map>
#include <vector>
#include "Word2VecCostFunctions.h"
#include "Word2VecDataset.h"
#include "StanfordSentimentDataset.h"
#include <mutex>
#include <thread>
#include <chrono>
#include <QDateTime>

template <class ScalarType>
void skipgram(const std::string &currentWord,
              unsigned int C,
              const std::vector<std::string> &contextWords,
              const std::map<std::string, unsigned int> &tokens,
              const std::vector<std::vector<ScalarType>> &inputVectors,
              const std::vector<std::vector<ScalarType>> &outputVectors,
              std::function<void(const std::vector<ScalarType> &, unsigned int, const std::vector<std::vector<ScalarType>> &, ScalarType &, std::vector<ScalarType> &, std::vector<std::vector<ScalarType>> &)> word2vecCostAndGradient,
              ScalarType &cost, std::vector<std::vector<ScalarType>> &inputGrad, std::vector<std::vector<ScalarType>> &outputGrad)
{

    std::vector<ScalarType> r = inputVectors[tokens.at(currentWord)];

    ScalarType overallCost = 0.0;
    std::vector<ScalarType> overallGradPred;
    overallGradPred.resize(r.size(), 0.0);

    outputGrad.resize(outputVectors.size());
    for(int i = 0; i< outputGrad.size(); ++i)
    {
        outputGrad[i].resize(r.size(), 0.0);
    }

    ScalarType eachCost = 0.0;
    cost = 0.0;
    std::vector<ScalarType> gradPred;
    std::vector<std::vector<ScalarType>> grad;

    for (int h = 0;h<contextWords.size();++h)
    {
        word2vecCostAndGradient(r, tokens.at(contextWords[h]), outputVectors, eachCost, gradPred, grad);
        cost += eachCost;

        for(int i = 0;i<gradPred.size();++i)
        {
            overallGradPred[i] += gradPred[i];
        }

        for(int i = 0;i<outputGrad.size();++i)
        {
            for(int e = 0;e<outputGrad[0].size();++e)
            {
                outputGrad[i][e] += grad[i][e];
            }
        }
    }

    inputGrad.resize(inputVectors.size());
    for(int i = 0; i< inputGrad.size(); ++i)
    {
        inputGrad[i].resize(r.size(), 0.0);
    }

    inputGrad[tokens.at(currentWord)] =  overallGradPred;

}

template<class ScalarType>
void CBOW(const std::string &currentWord,
          unsigned int C,
          const std::vector<std::string> &contextWords,
          const std::map<std::string, unsigned int> &tokens,
          const std::vector<std::vector<ScalarType>> &inputVectors,
          const std::vector<std::vector<ScalarType>> &outputVectors,
          std::function<void(const std::vector<ScalarType> &, unsigned int, const std::vector<std::vector<ScalarType>> &, ScalarType &, std::vector<ScalarType> &, std::vector<std::vector<ScalarType>> &)> word2vecCostAndGradient,
          ScalarType &cost, std::vector<std::vector<ScalarType>> &inputGrad, std::vector<std::vector<ScalarType>> &outputGrad)
{
    std::vector<ScalarType> r;
    r.resize(inputVectors[0].size(), 0);

    for(int i =0;i<contextWords.size();++i)
    {
        for(int e =0;e<r.size();++e)
        {
            r[e] += inputVectors[tokens.at(contextWords[i])][e];
        }
    }

    for(int e =0;e<r.size();++e)
    {
        r[e] /= C;
    }

    std::vector<ScalarType> gradPred;

    word2vecCostAndGradient(r, tokens.at(currentWord), outputVectors, cost, gradPred, outputGrad );

    inputGrad.resize(inputVectors.size());
    for(int i = 0; i< inputGrad.size(); ++i)
    {
        inputGrad[i].resize(r.size(), 0.0);
    }



    for(int i = 0;i<contextWords.size();++i)
    {
        for(int e = 0;e<gradPred.size();++e)
        {
            inputGrad[tokens.at(contextWords[i])][e] += gradPred[e] / C;
        }
    }
}

template<class ScalarType>
class WorkerThread
{
private:
    static std::mutex datasetMutex;
private:
    unsigned int m_threadId;
    unsigned int m_thisBatchSize;
    unsigned int m_overallBatchSize;
    const std::map<std::string, unsigned int> &m_tokens;
    const std::vector<std::vector<ScalarType>> &m_inputWordVectors;
    const std::vector<std::vector<ScalarType>> &m_outputWordVectors;
    Word2VecDataset &m_dataset;
    std::thread *m_thread;
    unsigned int m_C;
    std::function<void(const std::string &,
                                                    unsigned int,
                                                    const std::vector<std::string> &,
                                                    const std::map<std::string, unsigned int> &,
                                                    const std::vector<std::vector<ScalarType>> &,
                                                    const std::vector<std::vector<ScalarType>> &,
                                                    std::function<void(const std::vector<ScalarType> &, unsigned int, const std::vector<std::vector<ScalarType>> &, ScalarType &, std::vector<ScalarType> &, std::vector<std::vector<ScalarType>> &)> ,
                                                    ScalarType &, std::vector<std::vector<ScalarType>> &, std::vector<std::vector<ScalarType>> &)> m_word2VecModel;

    std::function<void(const std::vector<ScalarType> &, unsigned int, const std::vector<std::vector<ScalarType>> &, ScalarType &, std::vector<ScalarType> &, std::vector<std::vector<ScalarType>> &)> m_word2vecCostAndGradient;

public:
    ScalarType *m_cost;
    std::vector<std::vector<ScalarType>> *m_inGrad;
    std::vector<std::vector<ScalarType>> *m_outGrad;

    WorkerThread( unsigned int threadId,   std::function<void(const std::string &,
                                        unsigned int,
                                        const std::vector<std::string> &,
                                        const std::map<std::string, unsigned int> &,
                                        const std::vector<std::vector<ScalarType>> &,
                                        const std::vector<std::vector<ScalarType>> &,
                                        std::function<void(const std::vector<ScalarType> &, unsigned int, const std::vector<std::vector<ScalarType>> &, ScalarType &, std::vector<ScalarType> &, std::vector<std::vector<ScalarType>> &)> ,
                                        ScalarType &, std::vector<std::vector<ScalarType>> &, std::vector<std::vector<ScalarType>> &)> word2VecModel,
            unsigned int thisBatchSize, unsigned int overallBatchSize, const std::map<std::string, unsigned int> &tokens, unsigned int C, const std::vector<std::vector<ScalarType>> &inputWordVectors,    const std::vector<std::vector<ScalarType>> &outputWordVectors, Word2VecDataset &dataset,
                     std::function<void(const std::vector<ScalarType> &, unsigned int, const std::vector<std::vector<ScalarType>> &, ScalarType &, std::vector<ScalarType> &, std::vector<std::vector<ScalarType>> &)> word2vecCostAndGradient)
        : m_threadId(threadId),
          m_thisBatchSize(thisBatchSize),
          m_overallBatchSize(overallBatchSize),
          m_tokens(tokens),
          m_inputWordVectors(inputWordVectors),
          m_outputWordVectors(outputWordVectors),
          m_dataset(dataset),
          m_thread(NULL),
          m_C(C),
          m_word2VecModel(word2VecModel),
          m_word2vecCostAndGradient(word2vecCostAndGradient),
          m_cost(NULL),
          m_inGrad(NULL),
          m_outGrad(NULL)
    {
        m_cost = new ScalarType;
        *m_cost = 0.0;
        m_inGrad = new std::vector<std::vector<ScalarType>>;
        m_outGrad = new std::vector<std::vector<ScalarType>>;

        m_inGrad->resize(m_inputWordVectors.size());
        for(int i = 0;i<m_inGrad->size();++i)
        {
            (*m_inGrad)[i].resize(m_inputWordVectors[0].size(), 0.0);
        }

        m_outGrad->resize(m_outputWordVectors.size());
        for(int i =0;i<m_outGrad->size();++i)
        {
            (*m_outGrad)[i].resize(m_outputWordVectors[0].size(), 0.0);
        }
        //qDebug() << "debug cost " << m_threadId << m_cost;
    }

    WorkerThread(const WorkerThread &in)
        :m_threadId(in.m_threadId),
          m_thisBatchSize(in.m_thisBatchSize),
          m_overallBatchSize(in.m_overallBatchSize),
          m_tokens(in.m_tokens),
          m_inputWordVectors(in.m_inputWordVectors),
          m_outputWordVectors(in.m_outputWordVectors),
          m_dataset(in.m_dataset),
          m_thread(NULL),
          m_C(in.m_C),
          m_word2VecModel(in.m_word2VecModel),
          m_word2vecCostAndGradient(in.m_word2vecCostAndGradient),
          m_cost(in.m_cost),
          m_inGrad(in.m_inGrad),
          m_outGrad(in.m_outGrad)
    {
        /*m_cost = new ScalarType;
        *m_cost = *in.m_cost;
        m_inGrad = new std::vector<std::vector<ScalarType>>;
        m_outGrad = new std::vector<std::vector<ScalarType>>;

        m_inGrad->resize(m_inputWordVectors.size());
        for(int i = 0;i<m_inGrad->size();++i)
        {
            (*m_inGrad)[i].resize(m_inputWordVectors[0].size());

            for(int e = 0;e<m_inputWordVectors[0].size();++e)
            {
                (*m_inGrad)[i][e] = (*in.m_inGrad)[i][e];
            }
        }

        m_outGrad->resize(m_outputWordVectors.size());
        for(int i =0;i<m_outGrad->size();++i)
        {
            (*m_outGrad)[i].resize(m_outputWordVectors[0].size());

            for(int e = 0;e<m_outputWordVectors[0].size();++e)
            {
                (*m_outGrad)[i][e] = (*in.m_outGrad)[i][e];
            }
        }*/

        //qDebug() << "debug cost " << m_threadId << m_cost;
    }

    ~WorkerThread()
    {
        if (m_thread)
        {
            delete m_thread;
            m_thread = NULL;
        }

        /*if (m_cost)
        {
            delete m_cost;
            m_cost = NULL;
        }

        if (m_inGrad)
        {
            delete m_inGrad;
            m_inGrad = NULL;
        }

        if (m_outGrad)
        {
            delete m_outGrad;
            m_outGrad = NULL;
        }*/
    }

    void start()
    {
        m_thread = new std::thread(std::move(*this));

    }

    void join()
    {
        if (m_thread)
        {
            m_thread->join();
        }
    }

    void operator()()
    {
        for(int i =0; i<m_thisBatchSize;++i)
        {
            std::string centerWord;
            std::vector<std::string> context;
            unsigned int C1 = rand() % m_C + 1;
            WorkerThread<ScalarType>::datasetMutex.lock();
            m_dataset.getRandomContext(C1, centerWord, context);
            WorkerThread<ScalarType>::datasetMutex.unlock();
            ScalarType c;
            std::vector<std::vector<ScalarType>> gin;
            std::vector<std::vector<ScalarType>> gout;

            m_word2VecModel(centerWord, C1, context, m_tokens, m_inputWordVectors, m_outputWordVectors, m_word2vecCostAndGradient,  c, gin, gout);
            *m_cost += c / m_overallBatchSize;

            for(int h = 0;h<m_inGrad->size();++h)
            {
                for(int e=0;e<(*m_inGrad)[0].size();++e)
                {
                    (*m_inGrad)[h][e] += gin[h][e] / m_overallBatchSize;
                    (*m_outGrad)[h][e] += gout[h][e] / m_overallBatchSize;

                }
            }
        }

        //qDebug() << "one thread done" << m_threadId<< *m_cost << m_cost;
    }

};

template<class ScalarType>
std::mutex WorkerThread<ScalarType>::datasetMutex;

template<class ScalarType>
void word2VecSGDWrapper_multithread(std::function<void(const std::string &,
                                                unsigned int,
                                                const std::vector<std::string> &,
                                                const std::map<std::string, unsigned int> &,
                                                const std::vector<std::vector<ScalarType>> &,
                                                const std::vector<std::vector<ScalarType>> &,
                                                std::function<void(const std::vector<ScalarType> &, unsigned int, const std::vector<std::vector<ScalarType>> &, ScalarType &, std::vector<ScalarType> &, std::vector<std::vector<ScalarType>> &)> ,
                                                ScalarType &, std::vector<std::vector<ScalarType>> &, std::vector<std::vector<ScalarType>> &)> word2VecModel, const std::map<std::string, unsigned int> &tokens, const std::vector<std::vector<ScalarType>> &inputWordVectors, const std::vector<std::vector<ScalarType>> &outputWordVectors, Word2VecDataset &dataset, unsigned int C, std::function<void(const std::vector<ScalarType> &, unsigned int, const std::vector<std::vector<ScalarType>> &, ScalarType &, std::vector<ScalarType> &, std::vector<std::vector<ScalarType>> &)> word2vecCostAndGradient, ScalarType &cost, std::vector<std::vector<ScalarType>> &inGrad, std::vector<std::vector<ScalarType>> &outGrad)
{
    const int batchsize = 250;
    cost = 0.0;
    inGrad.resize(inputWordVectors.size());
    for(int i = 0;i<inGrad.size();++i)
    {
        inGrad[i].resize(inputWordVectors[0].size(), 0.0);
    }
    outGrad.resize(outputWordVectors.size());
    for(int i =0;i<outGrad.size();++i)
    {
        outGrad[i].resize(outputWordVectors[0].size(), 0.0);
    }

    unsigned int threadNum = std::thread::hardware_concurrency();

    unsigned int eachBatch = batchsize / threadNum;
    unsigned int remainder = batchsize % threadNum;

    std::vector<WorkerThread<ScalarType>*> threadPool;


    for(int i =0; i<threadNum; ++i)
    {
        unsigned int actualBatchSize = eachBatch;
        if (i<remainder)
        {
            actualBatchSize++;
        }

        if (actualBatchSize)
        {
            WorkerThread<ScalarType> *thread = new WorkerThread<ScalarType>(i, word2VecModel, actualBatchSize, batchsize, tokens,C, inputWordVectors, outputWordVectors, dataset, word2vecCostAndGradient);
            thread->start();
            threadPool.push_back(thread);
        }
        else
        {
            break;
        }
    }

    for(int i =0;i<threadPool.size();++i)
    {
        threadPool[i]->join();

        cost+= (*threadPool[i]->m_cost);

        for(int h = 0;h<inGrad.size();++h)
        {
            for(int e=0;e<inGrad[0].size();++e)
            {
                inGrad[h][e] += (*threadPool[i]->m_inGrad)[h][e] ;
                outGrad[h][e] += (*threadPool[i]->m_outGrad)[h][e] ;

            }
        }

        delete threadPool[i]->m_cost;
        delete threadPool[i]->m_inGrad;
        delete threadPool[i]->m_outGrad;
        delete threadPool[i];
        threadPool[i] = 0;
    }
}

template<class ScalarType>
void word2VecSGDWrapper(std::function<void(const std::string &,
                                                unsigned int,
                                                const std::vector<std::string> &,
                                                const std::map<std::string, unsigned int> &,
                                                const std::vector<std::vector<ScalarType>> &,
                                                const std::vector<std::vector<ScalarType>> &,
                                                std::function<void(const std::vector<ScalarType> &, unsigned int, const std::vector<std::vector<ScalarType>> &, ScalarType &, std::vector<ScalarType> &, std::vector<std::vector<ScalarType>> &)> ,
                                                ScalarType &, std::vector<std::vector<ScalarType>> &, std::vector<std::vector<ScalarType>> &)> word2VecModel, const std::map<std::string, unsigned int> &tokens, const std::vector<std::vector<ScalarType>> &inputWordVectors, const std::vector<std::vector<ScalarType>> &outputWordVectors, Word2VecDataset &dataset, unsigned int C, std::function<void(const std::vector<ScalarType> &, unsigned int, const std::vector<std::vector<ScalarType>> &, ScalarType &, std::vector<ScalarType> &, std::vector<std::vector<ScalarType>> &)> word2vecCostAndGradient, ScalarType &cost, std::vector<std::vector<ScalarType>> &inGrad, std::vector<std::vector<ScalarType>> &outGrad)
{
    const int batchsize = 250;
    cost = 0.0;
    inGrad.resize(inputWordVectors.size());
    for(int i = 0;i<inGrad.size();++i)
    {
        inGrad[i].resize(inputWordVectors[0].size(), 0.0);
    }
    outGrad.resize(outputWordVectors.size());
    for(int i =0;i<outGrad.size();++i)
    {
        outGrad[i].resize(outputWordVectors[0].size(), 0.0);
    }

    for(int i =0; i<batchsize;++i)
    {
        std::string centerWord;
        std::vector<std::string> context;
        unsigned int C1 = rand() % C + 1;
        dataset.getRandomContext(C1, centerWord, context);

        ScalarType c;
        std::vector<std::vector<ScalarType>> gin;
        std::vector<std::vector<ScalarType>> gout;

        word2VecModel(centerWord, C1, context, tokens, inputWordVectors, outputWordVectors, word2vecCostAndGradient,  c, gin, gout);
        cost += c / batchsize;

        for(int h = 0;h<inGrad.size();++h)
        {
            for(int e=0;e<inGrad[0].size();++e)
            {
                inGrad[h][e] += gin[h][e] / batchsize;
                outGrad[h][e] += gout[h][e] / batchsize;

            }
        }
    }
}

template<class ScalarType>
void normalizeRows(std::vector<std::vector<ScalarType>> &inGrad,std::vector<std::vector<ScalarType>> &outGrad)
{
    for(int i = 0;i<inGrad.size();++i)
    {
        ScalarType lenIn = 0.0;
        ScalarType lenOut = 0.0;
        for(int e=0;e<inGrad[0].size();++e)
        {
            lenIn += inGrad[i][e]*inGrad[i][e];
            lenOut += outGrad[i][e]*outGrad[i][e];
        }

        lenIn = std::sqrt(lenIn);
        lenOut = std::sqrt(lenOut);

        for(int e=0;e<inGrad[0].size();++e)
        {
            inGrad[i][e] /= lenIn;
            outGrad[i][e] /= lenOut;
        }
    }
}

template<class ScalarType>
void save(unsigned int iteration, std::vector<std::vector<ScalarType>> &inGrad, std::vector<std::vector<ScalarType>> &outGrad)
{
    QFile outputFile(QString("save_%1.dat").arg(iteration));
    outputFile.open(QFile::WriteOnly);

    unsigned int height = inGrad.size();
    unsigned int width = inGrad[0].size();

    outputFile.write((char*)&height, sizeof(height));
    outputFile.write((char*)&width, sizeof(width));

    for(int i =0;i<inGrad.size();++i)
    {
        outputFile.write((char*)&inGrad[i][0], sizeof(ScalarType) * inGrad[i].size());
    }

    for(int i =0;i<outGrad.size();++i)
    {
        outputFile.write((char*)&outGrad[i][0], sizeof(ScalarType) * outGrad[i].size());
    }

    outputFile.close();

}

template<class ScalarType>
void word2VecSGD(unsigned int offset, std::vector<std::vector<ScalarType>> &inGrad0, std::vector<std::vector<ScalarType>> &outGrad0, Word2VecDataset &dataset, ScalarType step, unsigned int iterations)
{
    // Anneal learning rate every several iterations
    const unsigned int ANNEAL_EVERY = 20000;
    const unsigned int SAVE_EVERY = 1000;

    std::vector<std::vector<ScalarType>> inGrad = inGrad0;
    std::vector<std::vector<ScalarType>> outGrad = outGrad0;

    const std::map<std::string, unsigned int>  &tokens = dataset.tokens();

    normalizeRows(inGrad, outGrad);

    QFile logFile(QString("logfile_%1.txt").arg(QDateTime::currentDateTime().toString()));
    logFile.open(QFile::Append);
    auto start = std::chrono::system_clock::now();
    unsigned int oldMinuteCount = 0;
    for(int i = 1; i< iterations+1; ++i)
    {
        ScalarType cost;
        std::vector<std::vector<ScalarType>> newInGrad;
        std::vector<std::vector<ScalarType>> newOutGrad;

        word2VecSGDWrapper_multithread<double>(skipgram<ScalarType>, tokens, inGrad, outGrad, dataset, 5, softmaxCostAndGradient<ScalarType>, cost, newInGrad, newOutGrad);

        for(int h = 0;h<inGrad.size();++h)
        {
            for(int e=0;e<inGrad[0].size();++e)
            {
                inGrad[h][e] -= step * newInGrad[h][e];
                outGrad[h][e] -= step * newOutGrad[h][e];
            }
        }

        normalizeRows(inGrad, outGrad);
        auto duration = std::chrono::duration_cast<std::chrono::minutes>(std::chrono::system_clock::now()-start);
        qDebug() << "iteration:" << offset + i << "cost" << cost << "duration" << duration.count();
        logFile.write(QString("%1 %2 %3\n").arg(offset+i).arg(cost).arg(duration.count()).toUtf8());
        logFile.flush();

        if (oldMinuteCount != duration.count())
        {
            oldMinuteCount = duration.count();

            double iterationPerMinute = i / (double) oldMinuteCount;

            qDebug() << "========== Estimated time: " << iterations / iterationPerMinute / 60.0 << "hours ==========";
        }

        if (i % SAVE_EVERY == 0)
        {
            save<ScalarType>(offset + i, inGrad, outGrad);
        }

        if (i % ANNEAL_EVERY == 0)
        {
            step *= 0.5;
        }
    }
    logFile.close();

    save<ScalarType>(iterations+1+offset, inGrad, outGrad);
}
#endif // WORD2VECMODELS

