#ifndef ELEMENTWISEPRODUCT_H
#define ELEMENTWISEPRODUCT_H

#include "Operator.h"

namespace FreeWill
{

    template<DeviceType DeviceUsed = CPU, typename DataType = float>
    class ElementwiseProduct : public Operator<DeviceUsed>
    {
    protected:
        using Operator<DeviceUsed>::input;
        using Operator<DeviceUsed>::output;

    public:
        ElementwiseProduct()
            :Operator<DeviceUsed>({"OperandA","OperandB"},{"Output"})
        {

        }

        virtual bool init() override
        {
            if (!input("OperandA") || !input("OperandB") || !output("Output"))
            {
                return false;
            }        

            if (input("OperandA")->shape() != input("OperandB")->shape())
            {
                return false;
            }

            if (input("OperandA")->shape() != output("Output")->shape())
            {
                return false;
            }

            return true;
        }

        virtual void evaluate() override
        {
            Tensor<DeviceUsed, DataType> *operandA = (Tensor<DeviceUsed, DataType> *) input("OperandA");
            Tensor<DeviceUsed, DataType> *operandB = (Tensor<DeviceUsed, DataType> *) input("OperandB");
            Tensor<DeviceUsed, DataType> *_output = (Tensor<DeviceUsed, DataType> *) output("Output");

            unsigned int size = operandA->shape().size();

            for(unsigned int i = 0; i<size; ++i)
            {
                (*_output)[i] = ((*operandA)[i]) * ((*operandB)[i]);
            }
        }

    };
}

#endif
