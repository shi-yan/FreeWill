#ifndef NEURONNETWORKLAYER_H
#define NEURONNETWORKLAYER_H

#include <cstring>
#include <vector>
#include <functional>
#include "ActivationFunctions.h"
#include <QDebug>

static int did = 0;

template<class ScalarType>
class NeuralNetworkLayer
{
private:
    unsigned int m_inputSize;
    unsigned int m_outputSize;
    ScalarType **m_weights;
    std::function<void (const std::vector<ScalarType>&, std::vector<ScalarType>&)> m_activationFunction;
    std::function<ScalarType(ScalarType)> m_activationFunctionDerivative;

public:
    NeuralNetworkLayer():
        m_inputSize(0),
        m_outputSize(0),
        m_weights(NULL),
        m_activationFunction(NULL),
        m_activationFunctionDerivative(NULL)
    {
    }

    NeuralNetworkLayer(unsigned int inputSize, unsigned int outputSize,
                       std::function<void (const std::vector<ScalarType>&, std::vector<ScalarType>&)> activationFunction,
                       std::function<ScalarType(ScalarType)> activationFunctionDerivative)
        :m_inputSize(inputSize),
          m_outputSize(outputSize),
          m_weights(NULL),
          m_activationFunction(activationFunction),
          m_activationFunctionDerivative(activationFunctionDerivative)
    {
        m_weights = new ScalarType*[m_outputSize];
        for(int i = 0; i < m_outputSize; ++i)
        {
            m_weights[i] = new ScalarType[m_inputSize + 1];
            memset(m_weights[i], 0, m_inputSize + 1);
        }
    }

    std::function<void (const std::vector<ScalarType>&, std::vector<ScalarType>&)> &getActivationFunction()
    {
        return m_activationFunction;
    }

    std::function<ScalarType(ScalarType)> &getActivationDerivative()
    {
        return m_activationFunctionDerivative;
    }

    unsigned int getOutputSize() const
    {
        return m_outputSize;
    }

    unsigned int getInputSize() const
    {
        return m_inputSize;
    }

    NeuralNetworkLayer(const NeuralNetworkLayer<ScalarType> &in)
        :
                m_inputSize(0),
                m_outputSize(0),
                m_weights(NULL),
                m_activationFunction(),
                m_activationFunctionDerivative()
    {
        m_inputSize = in.m_inputSize;
        m_outputSize = in.m_outputSize;

        m_weights = new ScalarType*[m_outputSize];

        for(int i = 0; i < m_outputSize; ++i)
        {
            m_weights[i] = new ScalarType[m_inputSize + 1];
            memcpy(m_weights[i], in.m_weights[i], m_inputSize + 1);
        }

       m_activationFunction = in.m_activationFunction;
       m_activationFunctionDerivative = in.m_activationFunctionDerivative;
    }

    void operator=(const NeuralNetworkLayer<ScalarType> &in)
    {
        if (m_weights)
        {
            qDebug() << m_weights;
            for(int i = 0; i< m_outputSize; ++i)
            {
                qDebug() << m_weights[i];
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
        m_activationFunctionDerivative = in.m_activationFunctionDerivative;
    }

    void randomWeights()
    {
        if (m_weights)
        {
            for(int i = 0; i < m_outputSize; ++i)
            {
                for(int e = 0; e < m_inputSize+1 ; ++e)
                {
                    m_weights[i][e] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
                }
            }
        }
    }

    bool forward(std::vector<ScalarType> &inputs, std::vector<ScalarType> &outputs)
    {
        if (inputs.size() != m_inputSize )
        {
            return false;
        }

        std::vector<ScalarType> z;
        z.resize(m_outputSize, 0.0);

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

    void calculateLayerGradient(std::vector<ScalarType> &inputActivation, std::function<ScalarType(ScalarType)> d, std::vector<ScalarType> &n, NeuralNetworkLayer<ScalarType> &w, std::vector<ScalarType> &newN)
    {
        for(int i = 0; i< m_outputSize; ++i)
        {
            m_weights[i][m_inputSize] = n[i];
            for(int e = 0; e< m_inputSize;++e)
            {
                m_weights[i][e] = inputActivation[e] * n[i];
            }
        }

        newN.resize(inputActivation.size());

        for(int i = 0; i< newN.size(); ++i)
        {
            newN[i] = 0;
            for(int e = 0; e<w.getOutputSize();++e)
            {
                newN[i] += w.m_weights[e][i] * n[e];
            }

            newN[i] *= d(inputActivation[i]);
        }
    }

    ~NeuralNetworkLayer()
    {
        if (m_weights)
        {
            for(int i = 0; i< m_outputSize; ++i)
            {
                delete [] m_weights[i];
                m_weights[i] = 0;
            }

            delete [] m_weights;
            m_weights = 0;
        }
    }
};

#endif // NEURONNETWORKLAYER_H
