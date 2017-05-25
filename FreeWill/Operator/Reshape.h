#ifndef RESHAPE_H
#define RESHAPE_H


#include "../DeviceSelection.h"
#include "Operator.h"
#include "../Tensor/Tensor.h"

#include "ElementwiseAdd_CUDA.h"

namespace FreeWill
{
    template<DeviceType DeviceUsed = DeviceType::CPU_NAIVE, typename DataType = float>
    class Reshape : public Operator<DeviceUsed>
    {
    protected:
        using Operator<DeviceUsed>::input;
        Shape m_newShape;
    public:
        Reshape(const Shape &newShape)
            :Operator<DeviceUsed>({"Tensor"}, {}),
              m_newShape(newShape)
        {
        }

        virtual bool init() override
        {
            FAIL_IF (input("Tensor") == nullptr);

            FAIL_IF (input("Tensor")->shape().size() != m_newShape.size());

            return true;
        }

        virtual void evaluate() override
        {
            Tensor<DeviceUsed, DataType> *tensor = input("Tensor")->template toType<DataType>();

            tensor->reshape(m_newShape);
        }
    };
}

#endif
