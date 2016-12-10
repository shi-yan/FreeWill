#ifndef RELUDERIVATIVE_H
#define RELUDERIVATIVE_H

#include "Operator.h"

namespace FreeWill
{
    template<DeviceType DeviceUsed = CPU, typename DataType = float>
    class ReLUDerivative : public Operator<DeviceUsed>
    {
    protected:
        using Operator<DeviceUsed>::input;
        using Operator<DeviceUsed>::output;

    public:
        ReLUDerivative()
            :Operator<DeviceUsed>({"Input"}, {"Output"})
        {}

        virtual bool init() override
        {
            if (!input("Input") || !output("Output"))
            {
                return false;
            }

            if (input("Input")->shape() != output("Output")->shape())
            {
                return false;
            }

            return true;
        }

        virtual void evaluate() override
        {
            // return in > 0 ? in : 0.0;
            Tensor<DeviceUsed, DataType> *_input = (Tensor<DeviceUsed, DataType> *) input("Input");
            Tensor<DeviceUsed, DataType> *_output = (Tensor<DeviceUsed, DataType> *) output("Output");

            unsigned int size = _input->shape().size();

            for(unsigned int i =0;i<size; ++i)
            {
                (*_output)[i] = (*_input)[i] > 0.0 ? (*_input)[i] : 0.0;
            }
        }
    };

}

#endif
