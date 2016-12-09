#ifndef RELUDERIVATIVE_H
#define RELUDERIVATIVE_H

#include "Operator.h"

namespace FreeWill
{
    template<DeviceType DeviceUsed = CPU, typename DeviceType = float>
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
        }

        virtual void evaluate() override
        {
        }
    };

}

#endif
