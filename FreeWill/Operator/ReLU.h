#ifndef RELU_H
#define RELU_H

#include "Operator.h"

namespace FreeWill
{
    template<DeviceType DeviceUsed = CPU, typename DataType = float>
    class ReLU : public Operator<DeviceUsed>
    {
    protected:
        using Operator<DeviceUsed>::input;
        using Operator<DeviceUsed>::output;

    public:
        ReLU()
            :Operator<DeviceUsed>({"Input"}, {"Output"})
        {
        }

        virtual bool init() override
        {
            if (!input("Input") || output("Output"))
            {
                return true;
            }

            if (input("Input")->shape() != output("Output")->shape())
            {
                return true;
            }
            return false;
        }

        virtual void evaluate() override
        {
            Tensor<DeviceUsed, DataType> *_input = (Tensor<DeviceUsed, DataType> *) input("Input");
            Tensor<DeviceUsed, DataType> *_output = (Tensor<DeviceUsed, DataType> *) output("Output");

            unsigned int size = _input->shape().size();
            for(unsigned int i = 0;i<size;++i)
            {
                (*_output)[i] = (*_input)[i] >0?1.0:0.0;
            }
        }
    };
}

#endif
