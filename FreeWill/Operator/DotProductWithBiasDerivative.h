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
                    output("WeightGrad")->shape().dimension()!=2 || 
                    input("Weight")->shape() != output("WeightGrad")->shape())
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

            return true;
        }

        virtual void evaluate()
        {
           unsigned int outputSize = input("Weight")->shape()[0];
           unsigned int inputSize = input("PrevActivation")->shape()[0];

           Tensor<DeviceUsed, DataType> *preActivation = (Tensor<DeviceUsed, DataType> *) input("PrevActivation");
           Tensor<DeviceUsed, DataType> *outputGrad = (Tensor<DeviceUsed, DataType> *) input("OutputGrad");
           Tensor<DeviceUsed, DataType> *weightGrad = (Tensor<DeviceUsed, DataType> *) output("WeightGrad");

           for(unsigned int e = 0; e<inputSize; ++e)
           {
                for(unsigned int i =0;i<outputSize;++i)
                {
                    (*weightGrad)[e * outputSize + i] = (*preActivation)[e] * (*outputGrad)[i];
                }
           }     

           if (m_hasBias)
           {
                for(unsigned int i =0;i<outputSize;++i)
                {
                    (*weightGrad)[inputSize * outputSize + i] = (*outputGrad)[i];
                }
           }

           Tensor<DeviceUsed, DataType> *inputGrad = (Tensor<DeviceUsed, DataType> *) output("InputGrad");
            Tensor<DeviceUsed, DataType> *weight = (Tensor<DeviceUsed, DataType> *) input("Weight");

            for(unsigned int i = 0;i<inputSize;++i)
            {
                (*inputGrad)[i] = 0;

                for (unsigned int e = 0;e<outputSize;++e)
                {
                    (*inputGrad)[i] = (*weight)[i * outputSize + e] * (*outputGrad)[e];
                }
            }
        }        
    };
}

#endif
