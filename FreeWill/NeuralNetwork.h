#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>
#include <QString>
#include <time.h>
#include "NeuralNetworkLayer.h"
#include "ActivationFunctions.h"
#include <QFile>

#include <QThread>
#include <QSemaphore>




template<class ScalarType>
class NeuralNetwork
{
public:
    std::vector<NeuralNetworkLayer<ScalarType>> m_layers;
    unsigned int m_inputSize;
    unsigned int m_outputSize;
    std::function<void (const std::vector<ScalarType>&, const std::vector<ScalarType>&, ScalarType&)> m_costFunction;
    std::function<void (const std::vector<ScalarType>&, const std::vector<ScalarType>&, std::vector<ScalarType>&)> m_costFunctionDerivative;

public:
    class TrainingData
    {
    private:
        std::vector<ScalarType> m_inputs;
        std::vector<ScalarType> m_outputs;

    public:
        TrainingData():m_inputs(), m_outputs()
        {}

        TrainingData(std::vector<ScalarType> &inputs, std::vector<ScalarType> &outputs)
            :m_inputs(inputs),
              m_outputs(outputs)
        {
        }

        TrainingData(const TrainingData &in)
            :m_inputs(in.m_inputs),
              m_outputs(in.m_outputs)
        {}

        void operator=(const TrainingData &in)
        {
            m_inputs = in.m_inputs;
            m_outputs = in.m_outputs;
        }

        const std::vector<ScalarType> &getInputs() const
        {
            return m_inputs;
        }

        const std::vector<ScalarType> &getOutputs() const
        {
            return m_outputs;
        }
    };

    typedef std::vector<NeuralNetwork<ScalarType>::TrainingData> MiniBatch;

    class NeuralNetworkThread : public QThread
    {
    public:
        NeuralNetwork<ScalarType> &m_network;
        std::vector<NeuralNetworkLayer<ScalarType>> m_gradientForBatch;
        const NeuralNetwork<ScalarType>::MiniBatch &m_miniBatch;
        int m_offset;
        int m_size;
        QSemaphore &m_semaphore;
        ScalarType m_cost;

    public:


        NeuralNetworkThread(    NeuralNetwork<ScalarType> &network,
                                const NeuralNetwork<ScalarType>::MiniBatch &miniBatch,
                                int offset,
                                int size,
                                QSemaphore &semaphore)
            :QThread(),
              m_network(network),
              m_miniBatch(miniBatch),
              m_offset(offset),
              m_size(size),
              m_semaphore(semaphore),
              m_gradientForBatch(),
              m_cost(0.0f)
        {

        }

        void run() override
        {
            m_cost = 0.0;

            for(size_t i = 0; i < m_network.m_layers.size(); ++i)
            {
                NeuralNetworkLayer<ScalarType> layer(m_network.m_layers[i].getInputSize(), m_network.m_layers[i].getOutputSize(), m_network.m_layers[i].getActivationFunction(), m_network.m_layers[i].getActivationDerivative());
                m_gradientForBatch.push_back(layer);
            }

            for(size_t b = m_offset; b< m_offset+m_size; ++b)
            {
                std::vector<NeuralNetworkLayer<ScalarType>> gradientForOneData;
                gradientForOneData.reserve(m_network.m_layers.size());
                for(size_t i = 0; i < m_network.m_layers.size(); ++i)
                {
                    NeuralNetworkLayer<ScalarType> layer(m_network.m_layers[i].getInputSize(), m_network.m_layers[i].getOutputSize(), m_network.m_layers[i].getActivationFunction(), m_network.m_layers[i].getActivationDerivative());
                    gradientForOneData.push_back(layer);
                }

                std::vector<std::vector<ScalarType>> activations;
                activations.push_back(m_miniBatch[b].getInputs());

                std::vector<ScalarType> *previousInput = &activations[activations.size() - 1];

                for(size_t i =0 ;i<m_network.m_layers.size();++i)
                {
                    std::vector<ScalarType> activation;
                    m_network.m_layers[i].forward(*previousInput, activation);
                    activations.push_back(activation);

                    previousInput = &activations[activations.size() - 1];
                }

                float costForOneData = 0.0;
                std::vector<ScalarType> derivativesWithRespectToOutputs;

                m_network.m_costFunction(*previousInput, m_miniBatch[b].getOutputs(), costForOneData);

                m_network.m_costFunctionDerivative(*previousInput, m_miniBatch[b].getOutputs(), derivativesWithRespectToOutputs);

                m_cost += costForOneData;
                std::vector<ScalarType> n = derivativesWithRespectToOutputs;
                std::vector<ScalarType> newN;

                for(int i = gradientForOneData.size() - 1; i >= 0; --i)
                {
                    gradientForOneData[i].calculateLayerGradient(activations[i], m_network.m_layers[i].getActivationDerivative(), n, m_network.m_layers[i], newN);
                    m_gradientForBatch[i].merge(gradientForOneData[i]);
                    n = newN;
                }
            }

            m_semaphore.release();
        }
    };


    NeuralNetwork()
        :m_layers(),
          m_inputSize(0),
          m_outputSize(0)
    {}

    ~NeuralNetwork(){}

    void init(unsigned int inputSize, unsigned int outputSize,
              const std::vector<unsigned int> &neuronCountsForAllLayers,
              std::function<void (const std::vector<ScalarType>&, std::vector<ScalarType>&)> activationForInnerLayers,
              std::function<void (const std::vector<ScalarType>&, std::vector<ScalarType>&)> activationForLastLayer,
              std::function<void (const std::vector<ScalarType>&, const std::vector<ScalarType>&, ScalarType&)> costFunction,
              std::function<void (const std::vector<ScalarType>&, const std::vector<ScalarType>&, std::vector<ScalarType>&)> costFunctionDerivitive)
    {
        m_inputSize = inputSize;
        m_outputSize = outputSize;
        int previousLayerSize = inputSize;
        m_costFunction = costFunction;
        m_costFunctionDerivative = costFunctionDerivitive;

        for(size_t i = 0; i < neuronCountsForAllLayers.size(); ++i)
        {
            unsigned int layerSize = neuronCountsForAllLayers[i];
            NeuralNetworkLayer<ScalarType> oneLayer(previousLayerSize, layerSize, activationForInnerLayers, sigmoidDerivative<ScalarType>);

            m_layers.push_back(oneLayer);
            previousLayerSize = layerSize;
        }

        NeuralNetworkLayer<ScalarType> lastLayer(previousLayerSize, outputSize, activationForLastLayer, sigmoidDerivative<ScalarType>);

        m_layers.push_back(lastLayer);
    }

    void randomWeights()
    {
        srand(time(NULL));

        for(size_t i = 0; i < m_layers.size(); ++i)
        {
            m_layers[i].randomWeights();
        }
    }

    void dumpWeights(const QString fileName, int count)
    {



        std::vector<ScalarType> output;
        for(size_t i = 0; i < m_layers.size(); ++i)
        {
            m_layers[i].flatten(output);
        }

        QFile file(QString(fileName).append(QString("_%1.sav").arg(count)));
        file.open(QIODevice::WriteOnly);
        file.write((char*)&output[0], sizeof(ScalarType) * output.size());
        file.close();
    }

    void assignWeights(const std::vector<ScalarType> &weights)
    {
        int offset = 0;
        for(size_t i = 0; i < m_layers.size(); ++i)
        {
            m_layers[i].assignWeights(weights, offset);
        }
    }

    void getResult(const std::vector<ScalarType> &inputs, std::vector<ScalarType> &outputs)
    {
        std::vector<std::vector<ScalarType>> activations;
        activations.push_back(inputs);

        std::vector<ScalarType> *previousInput = &activations[activations.size() - 1];

        for(size_t i =0 ;i < m_layers.size(); ++i)
        {
            std::vector<ScalarType> activation;
            m_layers[i].forward(*previousInput, activation);
            activations.push_back(activation);

            previousInput = &activations[activations.size() - 1];
        }

        outputs = *previousInput;
    }

    void forwardPropagate(const NeuralNetwork<ScalarType>::MiniBatch &miniBatch, ScalarType &cost, std::vector<NeuralNetworkLayer<ScalarType>> &gradient)
    {
        cost = 0.0;

        for(size_t i = 0; i < m_layers.size(); ++i)
        {
            NeuralNetworkLayer<ScalarType> layer(m_layers[i].getInputSize(), m_layers[i].getOutputSize(), m_layers[i].getActivationFunction(), m_layers[i].getActivationDerivative());
            gradient.push_back(layer);
        }

        for(size_t b = 0; b<miniBatch.size(); ++b)
        {
            std::vector<NeuralNetworkLayer<ScalarType>> gradientForOneData;
            gradientForOneData.reserve(m_layers.size());
            for(size_t i = 0; i < m_layers.size(); ++i)
            {
                NeuralNetworkLayer<ScalarType> layer(m_layers[i].getInputSize(), m_layers[i].getOutputSize(), m_layers[i].getActivationFunction(), m_layers[i].getActivationDerivative());
                gradientForOneData.push_back(layer);
            }

            std::vector<std::vector<ScalarType>> activations;
            activations.push_back(miniBatch[b].getInputs());

            std::vector<ScalarType> *previousInput = &activations[activations.size() - 1];

            for(size_t i =0 ;i<m_layers.size();++i)
            {
                std::vector<ScalarType> activation;
                m_layers[i].forward(*previousInput, activation);
                activations.push_back(activation);

                previousInput = &activations[activations.size() - 1];
            }

            ScalarType costForOneData = 0.0;
            std::vector<ScalarType> derivativesWithRespectToOutputs;

            m_costFunction(*previousInput, miniBatch[b].getOutputs(), costForOneData);

            m_costFunctionDerivative(*previousInput, miniBatch[b].getOutputs(), derivativesWithRespectToOutputs);

            cost += costForOneData;
            std::vector<ScalarType> n = derivativesWithRespectToOutputs;
            std::vector<ScalarType> newN;

            for(int i = gradientForOneData.size() - 1; i >= 0; --i)
            {
                gradientForOneData[i].calculateLayerGradient(activations[i], m_layers[i].getActivationDerivative(), n, m_layers[i], newN);
                gradient[i].merge(gradientForOneData[i]);
                n = newN;
            }
        }

        for(size_t i = 0; i < gradient.size(); ++i)
        {
            gradient[i].normalize(1.0 / miniBatch.size());
        }

        cost /= miniBatch.size();
    }

    void forwardPropagateParallel(int threadCount, const NeuralNetwork<ScalarType>::MiniBatch &miniBatch, ScalarType &cost, std::vector<NeuralNetworkLayer<ScalarType>> &gradient)
    {

        int batchSize = miniBatch.size() / threadCount;
        int batchRemain = miniBatch.size() % threadCount;

        int offset = 0;
        QSemaphore semaphore;

        std::vector<NeuralNetworkThread*> threadPool;

        for(int b = 0;b<threadCount;++b)
        {
            int currentBatchSize = batchSize + (b<batchRemain?1:0);

            NeuralNetworkThread *thread = new NeuralNetworkThread(*this, miniBatch, offset, currentBatchSize,semaphore );
            threadPool.push_back(thread);
            offset += currentBatchSize;

            thread->start();
        }

        semaphore.acquire(threadCount);

        cost = 0.0;

        for(size_t i = 0; i < m_layers.size(); ++i)
        {
            NeuralNetworkLayer<ScalarType> layer(m_layers[i].getInputSize(), m_layers[i].getOutputSize(), m_layers[i].getActivationFunction(), m_layers[i].getActivationDerivative());
            gradient.push_back(layer);
        }

        for (int e =0;e<threadPool.size();++e)
        {
            cost += threadPool[e]->m_cost;
            for(int i = threadPool[e]->m_gradientForBatch.size() - 1; i >= 0; --i)
            {
                gradient[i].merge( threadPool[e]->m_gradientForBatch[i]);
            }

            delete threadPool[e];
        }

        for(size_t i = 0; i < gradient.size(); ++i)
        {
            gradient[i].normalize(1.0 / miniBatch.size());
        }

        cost /= miniBatch.size();
    }

    void updateWeights(ScalarType rate, const std::vector<NeuralNetworkLayer<ScalarType>> &gradient)
    {
        for(size_t i = 0; i< m_layers.size(); ++i)
        {
            m_layers[i].updateWeights(rate, gradient[i]);
        }
    }
};

void testNeuralNetwork();

#endif // NEURALNETWORK_H
