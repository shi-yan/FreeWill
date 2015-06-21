#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>
#include <QString>
#include <time.h>
#include "NeuralNetworkLayer.h"
#include "ActivationFunctions.h"

template<class ScalarType>
class NeuralNetwork
{
private:
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

        for(int i = 0; i< neuronCountsForAllLayers.size(); ++i)
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

        for(int i = 0; i<m_layers.size(); ++i)
        {
            m_layers[i].randomWeights();
        }
    }

    void assignWeights(const std::vector<ScalarType> &weights)
    {
        int offset = 0;
        for(int i = 0; i<m_layers.size(); ++i)
        {
            m_layers[i].assignWeights(weights, offset);
        }
    }

    bool forwardPropagate(const NeuralNetwork<ScalarType>::MiniBatch &miniBatch, ScalarType &cost, std::vector<NeuralNetworkLayer<ScalarType>> &gradient)
    {
        cost = 0.0;

        for(int i = 0; i < m_layers.size(); ++i)
        {
            NeuralNetworkLayer<ScalarType> layer(m_layers[i].getInputSize(), m_layers[i].getOutputSize(), m_layers[i].getActivationFunction(), m_layers[i].getActivationDerivative());
            gradient.push_back(layer);
        }

        for(int b = 0; b<miniBatch.size(); ++b)
        {
            std::vector<NeuralNetworkLayer<ScalarType>> gradientForOneData;
            gradientForOneData.reserve(3);
            for(int i = 0; i < m_layers.size(); ++i)
            {
                NeuralNetworkLayer<ScalarType> layer(m_layers[i].getInputSize(), m_layers[i].getOutputSize(), m_layers[i].getActivationFunction(), m_layers[i].getActivationDerivative());
                gradientForOneData.push_back(layer);
            }

            std::vector<std::vector<ScalarType>> activations;
            activations.push_back(miniBatch[b].getInputs());

            std::vector<ScalarType> *previousInput = &activations[activations.size() - 1];

            for(int i =0 ;i<m_layers.size();++i)
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

            for(int i = gradientForOneData.size() - 1; i>=0; --i)
            {
                gradientForOneData[i].calculateLayerGradient(activations[i], m_layers[i].getActivationDerivative(), n, m_layers[i], newN);
                gradient[i].merge(gradientForOneData[i]);
                n = newN;
            }
        }

        for(int i = 0; i< gradient.size();++i)
        {
            gradient[i].normalize(1.0/miniBatch.size());
        }

        cost /= miniBatch.size();
    }
};

void testNeuralNetwork();

#endif // NEURALNETWORK_H
