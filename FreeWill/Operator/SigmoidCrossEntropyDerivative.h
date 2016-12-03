#ifndef SIGMOIDCROSSENTROPYDERIVATIVE_H
#define SIGMOIDCROSSENTROPYDERIVATIVE_H

#include "Operator.h"

namespace FreeWill
{
    template<DeviceType DeviceUsed = CPU, typename DataType = float>
    class SigmoidCrossEntropyDerivative : public Operator<DeviceUsed>
    {
    protected:
        using Operator<DeviceUsed>::input;
        using Operator<DeviceUsed>::output;

    public:
        SigmoidCrossEntropyDerivative()
        :Operator<DeviceUsed>({"Input", "Label"},{"Output"})
        {
        
        }

        virtual bool init() override
        {
            if (!input("Input") || !output("Output") || !input("Label"))
            {
                return false;
            }

            if ((input("Input")->shape().dimension() != 2 ) || 
                    (output("Output")->shape().dimension() != 2) || 
                    (input("Label")->shape().dimension() != 2))
            {
                return false;
            }

            if ((input("Input")->shape() != input("Label")->shape()) || (input("Input")->shape() != output("Output")->shape()))
            {
                return false;
            }

            return true;
        }

        virtual void evaluate() override
        {
            Tensor<DeviceUsed, DataType> *_input = (Tensor<DeviceUsed, DataType> *) input("Input");            
            Tensor<DeviceUsed, DataType> *_output = (Tensor<DeviceUsed, DataType> *) output("Output");
            Tensor<DeviceUsed, DataType> *_label = (Tensor<DeviceUsed, DataType> *) input("Label");

            unsigned int batchSize = _input->shape()[1];
            unsigned int vectorSize = _input->shape()[0];

            printf("batchSize %d vectorSize %d\n", batchSize, vectorSize);
            printf("%d, %d, %d\n", _input->shape().size(), _output->shape().size(), _label->shape().size());

            for(unsigned int e = 0;e<batchSize;++e)
            {
                for(unsigned int i = 0; i < vectorSize; ++i)
                {
                    (*_output)[e * vectorSize + i] = (*_input)[e * vectorSize + i] - (*_label)[e * vectorSize + i];
                }
            }
        }

            
    };
}

#endif
