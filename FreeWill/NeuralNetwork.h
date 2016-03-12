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
#include <thread>
#include <tuple>

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

    static void parallelTrainingKernel(NeuralNetwork<ScalarType> &network, const NeuralNetwork<ScalarType>::MiniBatch &miniBatch, int offset, int size, std::vector<NeuralNetworkLayer<ScalarType>> &gradientForBatch, ScalarType &cost)
    {
        cost = 0.0;

        for(size_t i = 0; i < network.m_layers.size(); ++i)
        {
            NeuralNetworkLayer<ScalarType> layer(network.m_layers[i].getInputSize(), network.m_layers[i].getOutputSize(), network.m_layers[i].getActivationFunction(), network.m_layers[i].getActivationDerivative());
            gradientForBatch.push_back(layer);
        }

        for(size_t b = offset; b< offset + size; ++b)
        {
            std::vector<NeuralNetworkLayer<ScalarType>> gradientForOneData;
            gradientForOneData.reserve(network.m_layers.size());
            for(size_t i = 0; i < network.m_layers.size(); ++i)
            {
                NeuralNetworkLayer<ScalarType> layer(network.m_layers[i].getInputSize(), network.m_layers[i].getOutputSize(), network.m_layers[i].getActivationFunction(), network.m_layers[i].getActivationDerivative());
                gradientForOneData.push_back(layer);
            }

            std::vector<std::vector<ScalarType>> activations;
            activations.push_back(miniBatch[b].getInputs());

            std::vector<ScalarType> *previousInput = &activations[activations.size() - 1];

            for(size_t i =0 ;i<network.m_layers.size();++i)
            {
                std::vector<ScalarType> activation;
                network.m_layers[i].forward(*previousInput, activation);
                activations.push_back(activation);

                previousInput = &activations[activations.size() - 1];
            }

            float costForOneData = 0.0;
            std::vector<ScalarType> derivativesWithRespectToOutputs;

            network.m_costFunction(*previousInput, miniBatch[b].getOutputs(), costForOneData);

            network.m_costFunctionDerivative(*previousInput, miniBatch[b].getOutputs(), derivativesWithRespectToOutputs);

            cost += costForOneData;
            std::vector<ScalarType> n = derivativesWithRespectToOutputs;
            std::vector<ScalarType> newN;

            for(int i = gradientForOneData.size() - 1; i >= 0; --i)
            {
                gradientForOneData[i].calculateLayerGradient(activations[i], network.m_layers[i].getActivationDerivative(), n, network.m_layers[i], newN);
                gradientForBatch[i].merge(gradientForOneData[i]);
                n = newN;
            }
        }
    }


    NeuralNetwork()
        :m_layers(),
          m_inputSize(0),
          m_outputSize(0)
    {}

    ~NeuralNetwork(){}

    void init(unsigned int inputSize, unsigned int outputSize,
              const std::vector<unsigned int> &neuronCountsForAllLayers,
              std::function<void (const std::vector<ScalarType>&, std::vector<ScalarType>&)> activationForInnerLayers,
              std::function<ScalarType (ScalarType in)> activationDerivativeForInnerLayers,
              std::function<void (const std::vector<ScalarType>&, std::vector<ScalarType>&)> activationForLastLayer,
              std::function<ScalarType (ScalarType in)> activationDerivativeForLastLayer,
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
            NeuralNetworkLayer<ScalarType> oneLayer(previousLayerSize, layerSize, activationForInnerLayers, activationDerivativeForInnerLayers);

            m_layers.push_back(oneLayer);
            previousLayerSize = layerSize;
        }

        NeuralNetworkLayer<ScalarType> lastLayer(previousLayerSize, outputSize, activationForLastLayer, activationDerivativeForLastLayer);

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

        std::vector<std::tuple<std::vector<NeuralNetworkLayer<ScalarType>>, ScalarType, std::thread>>  threadPool;
        threadPool.resize(threadCount);

        for(int b = 0;b<threadCount;++b)
        {
            int currentBatchSize = batchSize + (b<batchRemain?1:0);
            std::get<2>(threadPool[b]) = std::thread(NeuralNetwork<ScalarType>::parallelTrainingKernel, std::ref(*this), std::ref(miniBatch), offset, currentBatchSize, std::ref(std::get<0>(threadPool[b])), std::ref(std::get<1>(threadPool[b])));
            offset += currentBatchSize;
        }

        cost = 0.0;

        for(size_t i = 0; i < m_layers.size(); ++i)
        {
            NeuralNetworkLayer<ScalarType> layer(m_layers[i].getInputSize(), m_layers[i].getOutputSize(), m_layers[i].getActivationFunction(), m_layers[i].getActivationDerivative());
            gradient.push_back(layer);
        }

        for(int b = 0;b<threadCount;++b)
        {
            std::get<2>(threadPool[b]).join();
        }

        for (int e =0;e<threadPool.size();++e)
        {
            cost += std::get<1>(threadPool[e]);
            for(int i = std::get<0>(threadPool[e]).size() - 1; i >= 0; --i)
            {
                gradient[i].merge( std::get<0>(threadPool[e])[i]);
            }
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
