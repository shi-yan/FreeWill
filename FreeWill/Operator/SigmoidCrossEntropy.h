#ifndef SIGMOIDCROSSENTROPY_H
#define SIGMOIDCROSSENTROPY_H

#include "Operator.h"

namespace FreeWill
{
    template<DeviceType DeviceUsed = CPU, typename DataType = float>
    class SigmoidCrossEntropy : public Operator<DeviceUsed>
    {
    protected:
        using Operator<DeviceUsed>::input;
        using Operator<DeviceUsed>::output;

    public:
        SigmoidCrossEntropy()
            :Operator<DeviceUsed>({"Input", "Label"},{"Cost"})
        {}

        virtual bool init() override
        {
            if (!input("Input") || !output("Cost") || !input("Label"))
            {
                return false;
            }

            if (input("Input")->shape().dimension() != 2 || output("Cost")->shape().dimension() != 1)
            {
                return false;
            }

            if (input("Input")->shape() != input("Label")->shape())
            {
                return false;
            }

            if (input("Input")->shape()[1] != output("Cost")->shape()[0])
            {
                return false;
            }

            return true;
        }

        virtual void evaluate() override
        {
            Tensor<DeviceUsed, DataType> *_input = (Tensor<DeviceUsed, DataType> *) input("Input");
            Tensor<DeviceUsed, DataType> *_label = (Tensor<DeviceUsed, DataType> *) input("Label");

            Tensor<DeviceUsed, DataType> *_cost = (Tensor<DeviceUsed, DataType> *) output("Cost");

            unsigned int batchSize = _cost->shape()[0];
            unsigned int vectorSize = _input->shape()[0];

            for(unsigned int e = 0; e< batchSize; ++e)
            {
                (*_cost)[e] = 0;
                for(size_t i = 0; i < vectorSize; ++i)
                {
                    (*_cost)[e] += (*_label)[e * vectorSize + i]*log((*_input)[e * vectorSize + i]) 
                        + (1.0 - (*_label)[e*vectorSize +i])*log(1.0 - (*_input)[e*vectorSize+i]);
                }
            
                (*_cost)[e] *= -1.0;
            }
        }
    };
}

#endif
