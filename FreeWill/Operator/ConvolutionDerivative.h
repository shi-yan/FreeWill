#ifndef CONVOLUTIONDERIVATIVE_H
#define CONVOLUTIONDERIVATIVE_H

#include "Operator.h"

namespace FreeWill
{

    template<DeviceType DeviceUsed = CPU, typename DataType = float>
    class ConvolutionDerivative : public Operator<DeviceUsed>
    {
    protected:
        using Operator<DeviceUsed>::input;
        using Operator<DeviceUsed>::output;

    public:
        ConvolutionDerivative()
            :Operator<DeviceUsed>({"PrevActivation","OutputGrad","FeatureMap","Bias"},{"FeatureMapGrad","BiasGrad","InputGrad"})
        {
        }

        virtual bool init() override
        {}

        virtual void evaluate() override
        {
        }

    };
}

#endif
