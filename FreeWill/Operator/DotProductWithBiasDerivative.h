#ifndef DOTPRODUCTWITHBIASDERIVATIVE_H
#define DOTPRODUCTWITHBIASDERIVATIVE_H

#include "Operator.h"

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
            :Operator<DeviceUsed>({"PrevActivation", "OutputGrad", "Weight"},{"WeightGrad", "InputGrad"}),
             m_hasBias(hasBias)
        {
        }
        
        virtual bool init()
        {
            if (!input("PrevActivation") || !input("OutputGrad") || !input("Weight") || !output("WeightGrad") || !output("InputGrad"))
            {
               // qDebug() << input("PrevActivation");
               // qDebug() << input("OutputGrad");
               // qDebug() << input("Weight");
               // qDebug() << output("WeightGrad");
               // qDebug() << output("InputGrad");
                return false;
            }

            if (input("Weight")->shape().dimension()!=2 || 
                    output("WeightGrad")->shape().dimension()!=3 || 
                    input("Weight")->shape()[0] != output("WeightGrad")->shape()[0] ||
                    input("Weight")->shape()[1] != output("WeightGrad")->shape()[1])
            {
                return false;
            }

            if (output("InputGrad")->shape().dimension()!=2 || input("PrevActivation")->shape() != output("InputGrad")->shape() )
            {
                return false;
            }

            if (input("OutputGrad")->shape().dimension() != 2 || input("OutputGrad")->shape()[0] != input("Weight")->shape()[0])
            {
                return false;
            }

            if (input("PrevActivation")->shape().dimension() != 2 || 
                    (input("PrevActivation")->shape()[0] + (m_hasBias?1:0))!= input("Weight")->shape()[1])
            {
                return false;
            }

            unsigned int batchSize = input("PrevActivation")->shape()[1];

            if (output("WeightGrad")->shape()[2] != batchSize || 
                    output("InputGrad")->shape()[1] != batchSize || 
                    input("OutputGrad")->shape()[1]!=batchSize)
            {
                return false;
            }

            return true;
        }

        virtual void evaluate()
        {
           unsigned int outputSize = input("Weight")->shape()[0];
           unsigned int inputSize = input("PrevActivation")->shape()[0];
            unsigned int batchSize = input("PrevActivation")->shape()[1];

            unsigned int weightSize = outputSize * inputSize;

            if (m_hasBias)
            {
                weightSize += outputSize;
            }

           Tensor<DeviceUsed, DataType> *preActivation = (Tensor<DeviceUsed, DataType> *) input("PrevActivation");
           Tensor<DeviceUsed, DataType> *outputGrad = (Tensor<DeviceUsed, DataType> *) input("OutputGrad");
           Tensor<DeviceUsed, DataType> *weightGrad = (Tensor<DeviceUsed, DataType> *) output("WeightGrad");

            for(unsigned int b = 0;b<batchSize;++b)
            {

                for(unsigned int e = 0; e<inputSize; ++e)
                {
                    for(unsigned int i =0;i<outputSize;++i)
                    {
                        (*weightGrad)[ b*weightSize + e * outputSize + i] = (*preActivation)[b*inputSize + e] * (*outputGrad)[b*outputSize + i];
                    }
                }     

                if (m_hasBias)
                {
                    for(unsigned int i =0;i<outputSize;++i)
                    {
                        (*weightGrad)[b*weightSize + inputSize * outputSize + i] = (*outputGrad)[b*outputSize + i];
                    }
                }
            }

           Tensor<DeviceUsed, DataType> *inputGrad = (Tensor<DeviceUsed, DataType> *) output("InputGrad");
            Tensor<DeviceUsed, DataType> *weight = (Tensor<DeviceUsed, DataType> *) input("Weight");

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
    };
}

#endif
