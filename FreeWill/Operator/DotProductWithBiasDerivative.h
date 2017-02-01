#ifndef DOTPRODUCTWITHBIASDERIVATIVE_H
#define DOTPRODUCTWITHBIASDERIVATIVE_H

#include "Operator.h"
#include "../Context/Context.h"

namespace FreeWill
{
    template<DeviceType DeviceUsed = CPU, typename DataType = float>
    class DotProductWithBiasDerivative : public Operator<DeviceUsed>
    {
    protected:
        using Operator<DeviceUsed>::input;
        using Operator<DeviceUsed>::output;
        bool m_hasBias;

    public:
        DotProductWithBiasDerivative(bool hasBias = true)
            :Operator<DeviceUsed>({"InputActivation", "OutputDelta", "Weight"},{"WeightGrad", "BiasGrad", "InputDelta"}),
             m_hasBias(hasBias)
        {
        }
        
        virtual bool init()
        {
            FAIL_IF(!input("InputActivation") || !input("OutputDelta") || !input("Weight") || !output("WeightGrad") || !output("InputDelta"));
           
            if (m_hasBias)
            {
                FAIL_IF(!output("BiasGrad"));

                FAIL_IF(output("BiasGrad")->shape().dimension() != 1);
                
                FAIL_IF(output("BiasGrad")->shape()[0] != input("Weight")->shape()[0]);
            }

            FAIL_IF(input("Weight")->shape().dimension()!=2 || 
                    output("WeightGrad")->shape().dimension()!=2 || 
                    input("Weight")->shape()[0] != output("WeightGrad")->shape()[0] ||
                    input("Weight")->shape()[1] != output("WeightGrad")->shape()[1]);

            FAIL_IF(output("InputDelta")->shape().dimension()!=2 || input("InputActivation")->shape() != output("InputDelta")->shape());

            FAIL_IF(input("OutputDelta")->shape().dimension() != 2 || input("OutputDelta")->shape()[0] != input("Weight")->shape()[0]);
         

            FAIL_IF (input("InputActivation")->shape().dimension() != 2 || 
                    input("InputActivation")->shape()[0] != input("Weight")->shape()[1]);

            unsigned int batchSize = input("InputActivation")->shape()[1];

            FAIL_IF(output("InputDelta")->shape()[1] != batchSize || 
                    input("OutputDelta")->shape()[1] != batchSize);

            return true;
        }

        virtual void evaluate()
        {
           unsigned int outputSize = input("Weight")->shape()[0];
           unsigned int inputSize = input("InputActivation")->shape()[0];
           unsigned int batchSize = input("InputActivation")->shape()[1];

           printf("inputsize:%d, batchsize:%d, outputsize:%d\n", inputSize, batchSize, outputSize);
           //unsigned int weightSize = outputSize * inputSize;

           Tensor<DeviceUsed, DataType> *preActivation = (Tensor<DeviceUsed, DataType> *) input("InputActivation");
           Tensor<DeviceUsed, DataType> *outputGrad = (Tensor<DeviceUsed, DataType> *) input("OutputDelta");
           Tensor<DeviceUsed, DataType> *weightGrad = (Tensor<DeviceUsed, DataType> *) output("WeightGrad");
           Tensor<DeviceUsed, DataType> *inputGrad = (Tensor<DeviceUsed, DataType> *) output("InputDelta");
           Tensor<DeviceUsed, DataType> *weight = (Tensor<DeviceUsed, DataType> *) input("Weight");
           Tensor<DeviceUsed, DataType> *biasGrad = (Tensor<DeviceUsed, DataType> *) output("BiasGrad");

           (*weightGrad)[0] = 0;
           if constexpr ((DeviceUsed & (CPU | CPU_NAIVE)) != 0)
           {
                for(unsigned int b = 0;b<batchSize;++b)
                {
                    for(unsigned int e = 0; e<inputSize; ++e)
                    {
                        for(unsigned int i =0;i<outputSize;++i)
                        {
                            if (e*outputSize + i == 0)
                            {
                                printf("adding %f %f\n", (*preActivation)[b*inputSize + e] , (*outputGrad)[b*outputSize + i]);
                            } 
                            (*weightGrad)[ e * outputSize + i] += (*preActivation)[b*inputSize + e] * (*outputGrad)[b*outputSize + i];
                        }
                    }     

                    if (m_hasBias)
                    {
                        for(unsigned int i =0;i<outputSize;++i)
                        {
                            (*biasGrad)[i] += (*outputGrad)[b*outputSize + i];
                        }
                    }
                }

                for (unsigned int b = 0; b < batchSize; ++b)
                {
                    for(unsigned int i = 0;i<inputSize;++i)
                    {
                        (*inputGrad)[b *inputSize +  i] = 0;

                        for (unsigned int e = 0;e<outputSize;++e)
                        {
                            (*inputGrad)[b*inputSize + i] += (*weight)[i * outputSize + e] * (*outputGrad)[b*outputSize + e];
                        }
                    }
                }
           }
           else if constexpr ((DeviceUsed & (GPU | GPU_CUDA)) != 0)
           {
               DataType alpha = 1.0;
               DataType beta = 0.0;

                if constexpr (std::is_same<DataType, float>::value)
                {
                    RUN_CUBLAS(cublasSgemm(Context<DeviceUsed>::getSingleton().cublasHandle(), CUBLAS_OP_N, CUBLAS_OP_T,
                                           outputSize, inputSize, batchSize, &alpha, outputGrad->gpuDataHandle(), outputSize,
                                           preActivation->gpuDataHandle(), inputSize, 
                                           &beta, weightGrad->gpuDataHandle(), outputSize));
                    if (m_hasBias)
                    {
                        RUN_CUBLAS(cublasSgemv(Context<DeviceUsed>::getSingleton().cublasHandle(),CUBLAS_OP_N,
                                    outputSize, batchSize, &alpha, outputGrad->gpuDataHandle(),outputSize,
                                    Context<DeviceUsed>::getSingleton().template getSharedOneVector<DataType>(outputSize), 1, 
                                     &beta,biasGrad->gpuDataHandle(), 1));
                    }

                    RUN_CUBLAS(cublasSgemm(Context<DeviceUsed>::getSingleton().cublasHandle(), CUBLAS_OP_T, CUBLAS_OP_N,
                                           inputSize, batchSize, outputSize, &alpha, weight->gpuDataHandle(), outputSize,
                                           outputGrad->gpuDataHandle(), outputSize, 
                                           &beta, inputGrad->gpuDataHandle(), inputSize));

                }
                else if constexpr (std::is_same<DataType, double>::value)
                {
                     RUN_CUBLAS(cublasDgemm(Context<DeviceUsed>::getSingleton().cublasHandle(), CUBLAS_OP_N, CUBLAS_OP_T,
                                           outputSize, inputSize, batchSize, &alpha, outputGrad->gpuDataHandle(), outputSize,
                                           preActivation->gpuDataHandle(), inputSize, 
                                           &beta, weightGrad->gpuDataHandle(), outputSize));
                    if (m_hasBias)
                    {
                        RUN_CUBLAS(cublasDgemv(Context<DeviceUsed>::getSingleton().cublasHandle(),CUBLAS_OP_N,
                                    outputSize, batchSize, &alpha, outputGrad->gpuDataHandle(),outputSize,
                                    Context<DeviceUsed>::getSingleton().template getSharedOneVector<DataType>(outputSize), 1, 
                                     &beta,biasGrad->gpuDataHandle(), 1));
                    }

                    RUN_CUBLAS(cublasDgemm(Context<DeviceUsed>::getSingleton().cublasHandle(), CUBLAS_OP_T, CUBLAS_OP_N,
                                           inputSize, batchSize, outputSize, &alpha, weight->gpuDataHandle(), outputSize,
                                           outputGrad->gpuDataHandle(), outputSize, 
                                           &beta, inputGrad->gpuDataHandle(), inputSize));

                }                
           
           }
        }        
    };
}

#endif
