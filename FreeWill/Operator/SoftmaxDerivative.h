#ifndef SOFTMAX_H
#define SOFTMAX_H

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
        SoftmaxDerivative() : Operator<DeviceUsed>({"Input", "Label"},{"Loss"})
        {
        }

        virtual bool init() override
        {
            return true;
        }

        virtual void evaluate() override
        {
        
        }
    };
}

#endif
