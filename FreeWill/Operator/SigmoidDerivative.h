#ifndef SIGMOIDDERIVATIVE_H
#define SIGMOIDDERIVATIVE_H

#include "Operator.h"

namespace FreeWill
{

    template <DeviceType DeviceUsed = CPU, typename DataType = float>
    class SigmoidDerivative : public Operator<DeviceUsed>
    {
    protected:
        using Operator<DeviceUsed>::input;
        using Operator<DeviceUsed>::output;

    public:        
        SigmoidDerivative()
            :Operator<DeviceUsed>({"Input"},{"Output"})
        {}

        ~SigmoidDerivative(){}

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
            unsigned int size = input("Input")->shape().size();

            Tensor<DeviceUsed, DataType> *_input = (Tensor<DeviceUsed, DataType> *) input("Input");
            Tensor<DeviceUsed, DataType> *_output = (Tensor<DeviceUsed, DataType> *) output("Output");

            for (unsigned int i =0;i<size;++i)
            {
                (*_output)[i] = (*_input)[i] * (1.0 - (*_input)[i]);
            }
        }
    };
}

#endif
