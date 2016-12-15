#ifndef SOFTMAXDERIVATIVE_H
#define SOFTMAXDERIVATIVE_H

#include "Operator.h"

namespace FreeWill
{

    template<DeviceType DeviceUsed = CPU, typename DataType = float>
    class SoftmaxDerivative : public Operator<DeviceUsed>
    {
    protected:
        using Operator<DeviceUsed>::input;
        using Operator<DeviceUsed>::output;

    public:
        SoftmaxDerivative() : Operator<DeviceUsed>({"Output", "Label"},{"InputGrad"})
        {
        }

        virtual bool init() override
        {
            if (!input("Output") || !input("Label") || !output("InputGrad"))
            {
                return false;
            }

            if (input("Output")->shape() != output("InputGrad")->shape())
            {
                return false;
            }

            if (input("Output")->shape().dimension() != 2)
            {
                return false;
            }

            if (input("Label")->shape().dimension() != 1)
            {
                return false;
            }

            if (input("Output")->shape()[1] != input("Label")->shape()[0])
            {
                return false;
            }

            return true;
        }

        virtual void evaluate() override
        {
            Tensor<DeviceUsed, DataType> *_output = (Tensor<DeviceUsed, DataType> *) input("Output");
            Tensor<DeviceUsed, unsigned int> *_label = (Tensor<DeviceUsed, unsigned int> *) input("Label");
            Tensor<DeviceUsed, DataType> *_inputGrad = (Tensor<DeviceUsed, DataType> *) output("InputGrad");

            unsigned int batchSize = _output->shape()[1];

            for(unsigned int b = 0;b<batchSize;++b)
            {
                unsigned int vectorSize = _output->shape()[0];

                for(unsigned int i = 0;i<vectorSize;++i)
                {
                    (*_inputGrad)[b*vectorSize+ i] = (*_output)[b*vectorSize +i]; 
                }

                (*_inputGrad)[b*vectorSize + (*_label)[b]] -= 1.0;
            }        
        }
    };
}

#endif
