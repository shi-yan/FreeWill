#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>
#include <QString>
#include <time.h>
#include "NeuronNetworkLayer.h"

template<class ScalarType>
class NeuralNetwork
{
private:
    std::vector<NeuronNetworkLayer<ScalarType>> m_layers;
    unsigned int m_inputSize;
    unsigned int m_outputSize;
    std::function<void (const std::vector<ScalarType>&, const std::vector<ScalarType>&, ScalarType&)> m_costFunction;
    std::function<void (const std::vector<ScalarType>&, const std::vector<ScalarType>&, NeuronNetworkLayer<ScalarType>&)> m_costFunctionDerivitives;

public:
    class TrainingData
    {
    private:
        std::vector<ScalarType> m_inputs;
        std::vector<ScalarType> m_outputs;

    public:
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
              std::function<void (const std::vector<ScalarType>&, std::vector<ScalarType>&)> &activationForInnerLayers,
              std::function<void (const std::vector<ScalarType>&, std::vector<ScalarType>&)> &activationForLastLayer,
              std::function<void (const std::vector<ScalarType>&, const std::vector<ScalarType>&, ScalarType&)> &costFunction)
    {
        m_inputSize = inputSize;
        m_outputSize = outputSize;
        int previousLayerSize = inputSize;
        m_costFunction = costFunction;

        for(int i = 0; i< neuronCountsForAllLayers.size(); ++i)
        {
            unsigned int layerSize = neuronCountsForAllLayers[i];
            NeuronNetworkLayer<ScalarType> oneLayer(previousLayerSize, layerSize, activationForInnerLayers);
            m_layers.push_back(oneLayer);
            previousLayerSize = layerSize;
        }

        NeuronNetworkLayer<ScalarType> lastLayer(previousLayerSize, outputSize, activationForLastLayer);
        m_layers.push_back(lastLayer);
    }

    void randomWeights()
    {
        srand(time(NULL));

        foreach(NeuronNetworkLayer<ScalarType> &layer, m_layers)
        {
            layer.randomWeights();
        }
    }

    bool forwardPropagate(const NeuralNetwork<ScalarType>::MiniBatch &miniBatch, ScalarType &cost, std::vector<NeuronNetworkLayer<ScalarType>> &gradient)
    {
        cost = 0.0;

        foreach(NeuralNetwork<ScalarType>::TrainingData data, miniBatch)
        {
            std::vector<NeuronNetworkLayer<ScalarType>> gradientForOneData;
            for(int i = 0; i < m_layers.size(); ++i)
            {
                NeuronNetworkLayer<ScalarType> layer(m_layers[i].getInputSize(), m_layers[i].getOutputSize());
                gradientForOneData.push_back(layer);
            }

            std::vector<std::vector<ScalarType>> activations;
            std::vector<ScalarType> *previousInput = data.getInputs();

            for(int i =0 ;i<m_layers.size();++i)
            {
                std::vector<ScalarType> activation;
                m_layers[i].forward(*previousInput, activation);
                activations.push_back(activation);
                *previousInput = &activations[activations.size() - 1];
            }

            ScalarType costForOneData = 0.0;

            m_costFunction(*previousInput, data.getOutputs(), costForOneData);

            cost += costForOneData;

            for(int i = gradientForOneData.size() - 1; i>=0; --i)
            {
                if (i == gradientForOneData.size() - 1)
                {
                    //last layer
                    //m_gradientForLastLayer(*previousInput, data.getOutputs(), std::vector<ScalarType> &inputActivation, gradientForOneData[i]);
                }
                else
                {
                    //m_gradientForInnerLayer(previousInput, std::vector<ScalarType> &inputActivation, std::vector<ScalarType> &outputActivation, gradientForOneData[i]);
                }

                previousInput = &gradientForOneData[i];
            }
        }

        cost /= miniBatch.size();
    }
};

#endif // NEURALNETWORK_H
