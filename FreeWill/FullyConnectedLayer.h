#ifndef NEURONNETWORKLAYER_H
#define NEURONNETWORKLAYER_H

#include "FreeWill.h"
#include <cstring>
#include <vector>
#include <functional>
#include "ActivationFunctions.h"
#include "FullyConnectedLayerKernelGPU.h"
#include <QDebug>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>

namespace FreeWill
{
    template<bool UseGpu = __USE_CPU__, typename ScalarType = float>
    class FullyConnectedLayer
    {
    private:
        unsigned int m_inputSize;
        unsigned int m_outputSize;
        unsigned int m_batchSize;

        ScalarType *m_weights;
        ScalarType *m_gradients;
        ScalarType (*m_activation) (const ScalarType);

    public:
        FullyConnectedLayer()
            :m_inputSize(0),
              m_outputSize(0),
              m_batchSize(0),
              m_weights(nullptr),
              m_gradients(nullptr),
              m_activation(nullptr)
        {
        }

        __ON_CPU__ FullyConnectedLayer(unsigned int inputSize, unsigned int outputSize, unsigned batchSize,
                           Activation activation)
            :m_inputSize(inputSize),
             m_outputSize(outputSize),
             m_batchSize(batchSize),
             m_weights(nullptr),
             m_gradients(nullptr),
             m_activation(nullptr)
       {
            m_weights = new ScalarType[m_outputSize * (m_inputSize + 1)];
            m_gradients = new ScalarType[m_outputSize * (m_inputSize + 1)];
            memset(m_weights, 0, m_outputSize * (m_inputSize + 1) * sizeof(ScalarType));
            memset(m_gradients, 0, m_outputSize * (m_inputSize + 1) * sizeof(ScalarType));
            if(std::is_same<ScalarType, float>::value)
            {
                switch(activation)
                {
                case Sigmoid:
                    m_activation = fptr_sigmoid;
                    break;
                case Rectifier:
                    m_activation = fptr_rectifier;
                    break;
                case Tanh:
                    m_activation = fptr_tanh;
                    break;
                case None:
                default:
                    m_activation = fptr_noActivation;
                }
            }
            else if(std::is_same<ScalarType, double>::value)
            {
                switch(activation)
                {
                case Sigmoid:
                    m_activation = fptr_sigmoid_d;
                    break;
                case Rectifier:
                    m_activation = fptr_rectifier_d;
                    break;
                case Tanh:
                    m_activation = fptr_tanh_d;
                    break;
                case None:
                default:
                    m_activation = fptr_noActivation_d;
                }
            }
       }

       __ON_GPU__ FullyConnectedLayer(unsigned int inputSize, unsigned int outputSize, unsigned batchSize,
                           Activation activation)
            :m_inputSize(inputSize),
             m_outputSize(outputSize),
             m_batchSize(batchSize),
             m_weights(nullptr),
             m_gradients(nullptr),
             m_activation(nullptr)
       {
            cudaMalloc(&m_weights, m_outputSize * (m_inputSize + 1) * sizeof(ScalarType));
            cudaMalloc(&m_gradients, m_outputSize * (m_inputSize + 1) * sizeof(ScalarType));
            cudaMemset(m_weights, 0, m_outputSize * (m_inputSize + 1) * sizeof(ScalarType));
            cudaMemset(m_gradients, 0, m_outputSize * (m_inputSize + 1) * sizeof(ScalarType));


            if(std::is_same<ScalarType, float>::value)
            {
                m_activation = getActivationFuncFloat(activation);
            }
            else if(std::is_same<ScalarType, double>::value)
            {
                m_activation = getActivationFuncDouble(activation);
            }
       }

       FullyConnectedLayer(const FullyConnectedLayer<UseGpu, ScalarType> &in)
           :m_inputSize(0),
            m_outputSize(0),
            m_batchSize(0),
            m_weights(nullptr),
            m_gradients(nullptr),
            m_activation(nullptr)
       {
           *this = in;
       }

       __ON_CPU__ void operator=(const FullyConnectedLayer<UseGpu, ScalarType> &in)
       {
           if (m_weights)
           {
               delete [] m_weights;
           }

           if (m_gradients)
           {
               delete [] m_gradients;
           }

           m_inputSize = in.m_inputSize;
           m_outputSize = in.m_outputSize;

           m_weights = new ScalarType[m_outputSize * (m_inputSize + 1)];
           m_gradients = new ScalarType[m_outputSize * (m_inputSize + 1)];

           memcpy(m_weights, in.m_weights, m_outputSize * (m_inputSize + 1) * sizeof(ScalarType));
           memcpy(m_gradients, in.m_gradients, m_outputSize * (m_inputSize + 1) * sizeof(ScalarType));

           m_activation = in.m_activation;
       }

       __ON_GPU__ void operator=(const FullyConnectedLayer<UseGpu, ScalarType> &in)
       {
           if (m_weights)
           {
               cudaFree(m_weights);
               m_weights = nullptr;
           }

           if (m_gradients)
           {
               cudaFree(m_gradients);
               m_gradients = nullptr;
           }

           m_inputSize = in.m_inputSize;
           m_outputSize = in.m_outputSize;

           cudaMalloc(&m_weights, m_outputSize * (m_inputSize + 1) * sizeof(ScalarType));
           cudaMalloc(&m_gradients, m_outputSize * (m_inputSize + 1) * sizeof(ScalarType));

           cudaMemcpy(m_weights, in.m_weights, m_outputSize * (m_inputSize + 1) * sizeof(ScalarType), cudaMemcpyDeviceToDevice);
           cudaMemcpy(m_gradients, in.m_gradients, m_outputSize * (m_inputSize + 1) * sizeof(ScalarType), cudaMemcpyDeviceToDevice);


           m_activation = in.m_activation;
       }

       Activation getActivationFunction() const
       {
           return m_activation;
       }

       unsigned int getOutputSize() const
       {
           return m_outputSize;
       }

       unsigned int getInputSize() const
       {
           return m_inputSize;
       }

        unsigned int getBatchSize() const
        {
            return m_batchSize;
        }

        __ON_CPU__ void randomWeights()
        {
            if (m_weights)
            {
                for(size_t i = 0; i < m_outputSize * (m_inputSize + 1); ++i)
                {
                    m_weights[i] = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
                }
            }
        }

        __ON_GPU__ void randomWeights()
        {
            if (m_weights)
            {
                curandGenerator_t prng;
                curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_XORWOW);

                // Set the seed for the random number generator using the system clock
                curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());

                // Fill the array with random numbers on the device
                if (sizeof(ScalarType) == sizeof(float))
                {
                    curandGenerateUniform(prng, m_weights, m_outputSize * (m_inputSize + 1));
                }
                else if (sizeof(ScalarType) == sizeof(double))
                {
                    curandGenerateUniformDouble(prng, m_weights, m_outputSize * (m_inputSize * 1));
                }
                else
                {
                    //assert(0);
                }
            }
        }

        __ON_CPU__ void assignWeights(const std::vector<ScalarType> &sweights, int &offset)
        {
            if(m_weights)
            {
                for(size_t i = 0; i < m_outputSize * (m_inputSize + 1); ++i)
                {
                    m_weights[i] = sweights[offset++];
                }
            }
        }

        __ON_GPU__ void assignWeights(const std::vector<ScalarType> &sweights, int &offset)
        {
            if(m_weights)
            {
                cudaMemcpy(m_weights, &sweights[offset], m_outputSize * (m_inputSize + 1) * sizeof(ScalarType), cudaMemcpyHostToDevice);
            }
        }

        __ON_CPU__ void forward(ScalarType *inputs, ScalarType *outputs) const
        {
            ScalarType (*activation)(const ScalarType in) = nullptr;

            switch(m_activation)
            {
            case Sigmoid:
                activation = sigmoid<ScalarType>;
                break;
            case Rectifier:
                activation = rectifier<ScalarType>;
                break;
            case Tanh:
                activation = tanh<ScalarType>;
                break;
            case None:
            default:
                activation = noActivation<ScalarType>;
            }

            for(size_t i = 0; i < m_outputSize; ++i)
            {
                ScalarType value = 0.0;

                for(size_t e = 0; e < m_inputSize; ++e)
                {
                    value += m_weights[i * (m_inputSize+1) + e] * inputs[e];
                }

                value += m_weights[i*(m_inputSize+1) + m_inputSize];

                outputs[i] = activation(value);
            }
        }

        __ON_GPU__ void forward(ScalarType *inputs, ScalarType *outputs) const
        {
            FullyConnectedLayerKernelGPU(m_weights, inputs, outputs, m_activation);
        }

        __ON_CPU__
        void calculateLayerGradient(std::vector<ScalarType> &inputActivation,
                                    std::function<ScalarType(ScalarType)> d,
                                    std::vector<ScalarType> &n,
                                    FullyConnectedLayer<UseGpu, ScalarType> &w,
                                    std::vector<ScalarType> &newN) const
        {
            for(size_t i = 0; i < m_outputSize; ++i)
            {
                m_weights[i*m_outputSize + m_inputSize] = n[i];
                for(size_t e = 0; e < m_inputSize; ++e)
                {
                    m_weights[i*(m_inputSize+1) + e] = inputActivation[e] * n[i];
                }
            }

            newN.resize(inputActivation.size());

            for(size_t i = 0; i< newN.size(); ++i)
            {
                newN[i] = 0;
                for(size_t e = 0; e < w.getOutputSize(); ++e)
                {
                    newN[i] += w.m_weights[e*(m_inputSize+1) + i] * n[e];
                }

                newN[i] *= d(inputActivation[i]);
            }
        }

        __ON_GPU__
        void calculateLayerGradient(std::vector<ScalarType> &inputActivation, std::function<ScalarType(ScalarType)> d, std::vector<ScalarType> &n, FullyConnectedLayer<UseGpu, ScalarType> &w, std::vector<ScalarType> &newN) const
        {
            for(size_t i = 0; i < m_outputSize; ++i)
            {
                m_weights[i*m_outputSize + m_inputSize] = n[i];
                for(size_t e = 0; e < m_inputSize; ++e)
                {
                    m_weights[i*(m_inputSize+1) + e] = inputActivation[e] * n[i];
                }
            }

            newN.resize(inputActivation.size());

            for(size_t i = 0; i< newN.size(); ++i)
            {
                newN[i] = 0;
                for(size_t e = 0; e < w.getOutputSize(); ++e)
                {
                    newN[i] += w.m_weights[e*(m_inputSize+1) + i] * n[e];
                }

                newN[i] *= d(inputActivation[i]);
            }
        }

        __ON_CPU__
        void merge(const FullyConnectedLayer &in)
        {
            if (m_inputSize == in.m_inputSize && m_outputSize == in.m_outputSize)
            {
                for(size_t i = 0; i < m_outputSize; ++i)
                {
                    m_weights[i*(m_inputSize+1) + m_inputSize] += in.m_weights[i*(m_inputSize+1) + m_inputSize];
                    for(size_t e = 0; e < m_inputSize; ++e)
                    {
                        m_weights[i*(m_inputSize+1) + e] += in.m_weights[i*(m_inputSize+1) + e];
                    }
                }
            }
            else
            {
                qDebug() << "merge error";
            }
        }

        __ON_GPU__
        void merge(const FullyConnectedLayer &in)
        {
            if (m_inputSize == in.m_inputSize && m_outputSize == in.m_outputSize)
            {
                for(size_t i = 0; i < m_outputSize; ++i)
                {
                    m_weights[i*(m_inputSize+1) + m_inputSize] += in.m_weights[i*(m_inputSize+1) + m_inputSize];
                    for(size_t e = 0; e < m_inputSize; ++e)
                    {
                        m_weights[i*(m_inputSize+1) + e] += in.m_weights[i*(m_inputSize+1) + e];
                    }
                }
            }
            else
            {
                //qDebug() << "merge error";
            }
        }

        /*void normalize(ScalarType co)
        {
            for(size_t i = 0; i < m_outputSize; ++i)
            {
                for(size_t e = 0; e < m_inputSize + 1; ++e)
                {
                    m_weights[i*(m_inputSize+1)+ e] *= co;
                }
            }
        }*/

        void flatten(std::vector<ScalarType> &output) const
        {
            for(size_t i = 0; i < m_outputSize; ++i)
            {
                for(size_t e = 0; e < m_inputSize + 1; ++e)
                {
                    output.push_back(m_weights[i* (m_inputSize+1) + e]);
                }
            }
        }

        void display() const
        {
            qDebug() << "============= gradient ========";
            for(int i = 0; i < m_outputSize; ++i)
            {
                for(int e = 0; e < m_inputSize + 1; ++e)
                {
                    qDebug("%f ", m_weights[i*(m_inputSize+1) + e]) ;
                }

                qDebug()<<" ";
            }
            qDebug() << "============= gradient ========";
        }

        void updateWeights(ScalarType rate, const FullyConnectedLayer &gradient)
        {
            for(size_t i = 0; i < m_outputSize; ++i)
            {
                for(size_t e = 0; e < m_inputSize + 1; ++e)
                {
                    m_weights[i*(m_inputSize+1)+e] -= rate * gradient.m_weights[i*(m_inputSize+1) + e];
                }
            }
        }


        ~FullyConnectedLayer()
        {
            cleanup();
        }


    private:
        __ON_CPU__
        void cleanup()
        {
            if (m_weights)
            {
                delete [] m_weights;
                m_weights = nullptr;
            }

            if (m_gradients)
            {
                delete [] m_gradients;
                m_gradients = nullptr;
            }
        }

        __ON_GPU__
        void cleanup()
        {
            if (m_weights)
            {
                cudaFree(m_weights);
                m_weights = nullptr;
            }

            if (m_gradients)
            {
                cudaFree(m_gradients);
                m_gradients = nullptr;
            }
        }
    };
}

#endif // NEURONNETWORKLAYER_H
