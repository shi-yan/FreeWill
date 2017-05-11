#ifndef DUPLICATE_H
#define DUPLICATE_H

#include "Operator.h"

namespace FreeWill
{
    template<DeviceType DeviceUsed = DeviceType::CPU_NAIVE, typename DataType = float>
    class Duplicate : public Operator<DeviceUsed>
    {
    protected:
        using Operator<DeviceUsed>::input;
        using Operator<DeviceUsed>::output;

    public:
        Duplicate()
            :Operator<DeviceUsed>({"From"}, {"To"})
        {
        }

        virtual bool init() override
        {
            FAIL_IF (input("From") == nullptr);

            FAIL_IF (output("To") == nullptr);

            FAIL_IF (input("From")->shape().size() != output("To")->shape().size());

            return true;
        }

        virtual void evaluate() override
        {

        }
    };

}

#endif
