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
        using Operator<DeviceUsed>::m_deviceId;
        Shape m_newShape;
    public:
        Reshape(const Shape &newShape, unsigned int deviceId = 0)
            :Operator<DeviceUsed>({"Tensor"}, {}, deviceId),
              m_newShape(newShape)
        {
        }

        virtual bool init() override
        {
            CHECK_GPU;

            FAIL_IF (input("Tensor") == nullptr);

            FAIL_IF (input("Tensor")->shape().size() != m_newShape.size());

            return true;
        }

        virtual void evaluate() override
        {
            CHECK_GPU;

            Tensor<DeviceUsed, DataType> *tensor = input("Tensor")->template toType<DataType>();

            tensor->reshape(m_newShape);
        }
    };
}

#endif
