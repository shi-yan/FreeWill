#ifndef NEURONNETWORKLAYER_H
#define NEURONNETWORKLAYER_H

#include <cstring>
#include <vector>
#include <functional>
#include "ActivationFunctions.h"

template<class ScalarType>
class NeuronNetworkLayer
{
private:
    unsigned int m_inputSize;
    unsigned int m_outputSize;

    ScalarType **m_weights;
    std::function<void (const std::vector<ScalarType>&, std::vector<ScalarType>&)> m_activationFunction;

public:
    NeuronNetworkLayer(unsigned int inputSize, unsigned int outputSize,
                       std::function<void (const std::vector<ScalarType>&, std::vector<ScalarType>&)> &activationFunction = noActivation)
        :m_inputSize(inputSize),
          m_outputSize(outputSize),
          m_weights(NULL),
          m_activationFunction(activationFunction)
    {
        m_weights = new ScalarType*[m_outputSize];
        for(int i = 0; i < m_outputSize; ++i)
        {
            m_weights[i] = new ScalarType[m_inputSize + 1];
            memset(m_weights[i], 0, m_inputSize + 1);
        }
    }

    unsigned int getOutputSize() const
    {
        return m_outputSize;
    }

    unsigned int getInputSize() const
    {
        return m_inputSize;
    }

    NeuronNetworkLayer(const NeuronNetworkLayer<ScalarType> &in)
    {
        *this = in;
    }

    void operator=(const NeuronNetworkLayer<ScalarType> &in)
    {
        if (m_weights)
        {
            for(int i = 0; i< m_outputSize; ++i)
            {
                delete [] m_weights[i];
            }
            delete [] m_weights;
        }

        m_inputSize = in.m_inputSize;
        m_outputSize = in.m_outputSize;

        m_weights = new ScalarType*[m_outputSize];
        for(int i = 0; i < m_outputSize; ++i)
        {
            m_weights[i] = new ScalarType[m_inputSize + 1];
            memcpy(m_weights[i], in.m_weights[i], m_inputSize + 1);
        }

        m_activationFunction = in.m_activationFunction;
    }

    void randomWeights()
    {
        if (m_weights)
        {
            for(int i = 0; i < m_outputSize; ++i)
            {
                for(int e = 0; e < m_inputSize + 1; ++e)
                {
                    m_weights[i][e] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
                }
            }
        }
    }

    bool forward(std::vector<ScalarType> &inputs, std::vector<ScalarType> &outputs)
    {
        if (inputs.size() != m_inputSize || outputs.size() != m_outputSize)
        {
            return false;
        }

        std::vector<ScalarType> z;
        z.resize(outputs.size(), 0.0);

        for(int i = 0; i < m_outputSize; ++i)
        {
            for(int e = 0; e < m_inputSize; ++e)
            {
                z[i] += m_weights[i][e] * inputs[e];
            }

            z[i] += m_weights[i][m_inputSize];
        }

        m_activationFunction(z, outputs);
        return true;
    }

    ~NeuronNetworkLayer()
    {
        for(int i = 0; i< m_outputSize; ++i)
        {
            delete [] m_weights[i];
        }

        delete [] m_weights;
    }
};

#endif // NEURONNETWORKLAYER_H
