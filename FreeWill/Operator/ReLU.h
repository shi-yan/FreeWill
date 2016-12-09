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
        {}

        virtual bool init() override
        {}

        virtual void evaluate()
        {}
    };
}

#endif
