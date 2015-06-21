#ifndef NEURONNETWORKLAYER_H
#define NEURONNETWORKLAYER_H

#include <cstring>
#include <vector>
#include <functional>
#include "ActivationFunctions.h"
#include <QDebug>


static double sweights[] = { 0.35778736,  0.56078453,  1.08305124,  1.05380205, -1.37766937, -0.93782504,
  0.51503527,  0.51378595,  0.51504769 , 3.85273149 , 0.57089051,  1.13556564,
  0.95400176,  0.65139125, -0.31526924 , 0.75896922, -0.77282521, -0.23681861,
 -0.48536355,  0.08187414,  2.31465857, -1.86726519,  0.68626019 ,-1.61271587,
 -0.47193187,  1.0889506 ,  0.06428002, -1.07774478, -0.71530371 , 0.67959775,
 -0.73036663,  0.21645859,  0.04557184, -0.65160035,  2.14394409 , 0.63391902,
 -2.02514259,  0.18645431, -0.66178646,  0.85243333, -0.79252074 ,-0.11473644,
  0.50498728,  0.86575519, -1.20029641, -0.33450124, -0.47494531 ,-0.65332923,
  1.76545424,  0.40498171, -1.26088395,  0.91786195,  2.1221562  , 1.03246526,
 -1.51936997, -0.48423407,  1.26691115, -0.70766947,  0.44381943 , 0.77463405,
 -0.92693047, -0.05952536, -3.24126734, -1.02438764, -0.25256815 ,-1.24778318,
  1.6324113 , -1.43014138, -0.44004449,  0.13074058,  1.44127329 ,-1.43586215,
  1.16316375,  0.01023306, -0.98150865,  0.46210347,  0.1990597  ,-0.60021688,
  0.06980208, -0.3853136 ,  0.11351735,  0.66213067,  1.58601682 ,-1.2378155,
  2.13303337, -1.9520878 , -0.1517851 ,  0.58831721,  0.28099187 ,-0.62269952,
 -0.20812225, -0.49300093, -0.58936476,  0.8496021 ,  0.35701549 ,-0.6929096,
  0.89959988,  0.30729952,  0.81286212,  0.62962884, -0.82899501 ,-0.56018104,
  0.74729361,  0.61037027, -0.02090159,  0.11732738,  1.2776649  ,-0.59157139,
  0.54709738, -0.20219265, -0.2176812 ,  1.09877685,  0.82541635 , 0.81350964,
  1.30547881};

static int weightc=0;


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
            memset(m_weights[i], 0, (m_inputSize + 1) * sizeof(ScalarType));
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
            memcpy(m_weights[i], in.m_weights[i], (m_inputSize + 1) * sizeof(ScalarType));
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
            memcpy(m_weights[i], in.m_weights[i], (m_inputSize + 1) * sizeof(ScalarType));
        }

        m_activationFunction = in.m_activationFunction;
        m_activationFunctionDerivative = in.m_activationFunctionDerivative;
    }

    void randomWeights()
    {
        if (weightc == 115)
            weightc = 0;

        if (m_weights)
        {
            for(int i = 0; i < m_outputSize; ++i)
            {
                for(int e = 0; e < m_inputSize+1 ; ++e)
                {
                    m_weights[i][e] = sweights[weightc++];
                            //= static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
                }
            }
        }
    }

    void assignWeights(const std::vector<ScalarType> &sweights, int &offset)
    {
        if(m_weights)
        {
            for(int i = 0; i < m_outputSize; ++i)
            {
                for(int e = 0; e < m_inputSize+1 ; ++e)
                {
                    m_weights[i][e] = sweights[offset ++ ];
                            //= static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
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
                qDebug( ) << "w" <<  m_weights[i][e];
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

    void merge(const NeuralNetworkLayer &in)
    {
        if (m_inputSize == in.m_inputSize && m_outputSize == in.m_outputSize)
        {
            for(int i = 0; i< m_outputSize; ++i)
            {
                m_weights[i][m_inputSize] += in.m_weights[i][m_inputSize];
                for(int e = 0; e< m_inputSize; ++e)
                {
                    m_weights[i][e] += in.m_weights[i][e];
                }
            }
        }
        else
        {
            qDebug() << "merge error";
        }
    }

    void normalize(ScalarType co)
    {
        for(int i = 0; i< m_outputSize; ++i)
        {
            for(int e = 0; e< m_inputSize + 1; ++e)
            {
                m_weights[i][e] *= co;
            }
        }
    }

    void flatten(std::vector<ScalarType> &output)
    {
        for(int i = 0; i< m_outputSize; ++i)
        {
            for(int e = 0; e< m_inputSize + 1; ++e)
            {
                output.push_back( m_weights[i][e]);
            }
        }
    }

    void display()
    {
        qDebug() << "============= gradient ========";
        for(int i = 0; i< m_outputSize; ++i)
        {
            for(int e = 0; e< m_inputSize + 1; ++e)
            {
                qDebug("%f ", m_weights[i][e]) ;
            }

            qDebug()<<" ";
        }
        qDebug() << "============= gradient ========";

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
