#ifndef DOTPRODUCTWITHBIAS_H
#define DOTPRODUCTWITHBIAS_H

#include "Operator.h"

namespace FreeWill
{
    template<DeviceType DeviceUsed = CPU, typename DataType = float>
    class DotProductWithBias : public Operator<DeviceUsed>
    {
    protected:
        using Operator<DeviceUsed>::input;
        using Operator<DeviceUsed>::output;
        bool m_hasBias;
    public:
        DotProductWithBias(bool hasBias = true)
            :Operator<DeviceUsed>({"Input", "Weight"},{"Output"}),
            m_hasBias(hasBias)
        {
                
        }

        virtual bool init()
        {
            if(input("Input")==0 || input("Weight")==0 || output("Output") == 0)
            {
                return false;
            }

            if ((input("Input")->shape().dimension() != 2) || 
                    (input("Weight")->shape().dimension() !=2) || 
                    (output("Output")->shape().dimension() != 2))
            {
                return false;
            }

            unsigned int batchSize = input("Input")->shape()[1];
            unsigned int inputSize = input("Input")->shape()[0];
            unsigned int outputSize = output("Output")->shape()[0];

            if (batchSize != output("Output")->shape()[1] || batchSize == 0)
            {
                return false;
            }

            
            if(input("Weight")->shape()[1] != inputSize + (m_hasBias?1:0))
            {
                return false;
            }

            if (input("Weight")->shape()[0]!= outputSize
                    || output("Output")->shape()[1] != batchSize)
            {
                return false;
            }
            
            return true;
        }

        virtual void evaluate()
        {
            unsigned int batchSize = input("Input")->shape()[1];
            unsigned int inputSize = input("Input")->shape()[0];
            unsigned int outputSize = output("Output")->shape()[0];

            Tensor<DeviceUsed, DataType> *_input = (Tensor<DeviceUsed, DataType> *) input("Input");
            Tensor<DeviceUsed, DataType> *_weight = (Tensor<DeviceUsed, DataType> *) input("Weight");
            Tensor<DeviceUsed, DataType> *_output = (Tensor<DeviceUsed, DataType> *) output("Output");

            for(unsigned int b = 0; b < batchSize; ++b)
            {
                for(unsigned int o =0; o<outputSize;++o)
                {
                    (*_output)[b * outputSize + o] = 0;
                    for(unsigned int i = 0; i< inputSize; ++i)
                    {
                        (*_output)[b * outputSize + o] += (*_weight)[i * outputSize + o] * (*_input)[b* inputSize + i];
                    }

                    if (m_hasBias)
                    {
                        (*_output)[b * outputSize + o] += (*_weight)[inputSize * outputSize + o];
                    }
                }
            }        
        }
    };
}


#endif
