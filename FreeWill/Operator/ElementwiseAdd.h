#ifndef ELEMENTWISEADD_H
#define ELEMENTWISEADD_H

#include "../DeviceSelection.h"
#include "Operator.h"
#include "../Tensor/Tensor.h"

#include "ElementwiseAdd_CUDA.h"

namespace FreeWill
{
    template<DeviceType DeviceUsed = CPU_NAIVE, typename DataType = float>
    class ElementwiseAdd : public Operator<DeviceUsed>
    {
    protected:
        DataType m_rate;
        using Operator<DeviceUsed>::input;
        using Operator<DeviceUsed>::output;
   public:
        ElementwiseAdd(DataType rate = 1.0f)
            :Operator<DeviceUsed>({"OperandA", "OperandB"}, {"Result"}),
            m_rate(rate)
        {
        }

        void setRate(DataType rate)
        {
            m_rate = rate;
        }

        virtual bool init() override
        {
            FAIL_IF (input("OperandA") == nullptr);

            FAIL_IF (input("OperandB") == nullptr);

            FAIL_IF (output("Result") == nullptr);

            FAIL_IF (input("OperandA")->shape().size() != output("Result")->shape().size());

            FAIL_IF (input("OperandB")->shape().size() != output("Result")->shape().size());

            return true;
        }

        virtual void evaluate() override
        {

            Tensor<DeviceUsed, DataType> *result = output("Result")->template toType<DataType>();
            Tensor<DeviceUsed, DataType> *operandA = input("OperandA")->template toType<DataType>();
            Tensor<DeviceUsed, DataType> *operandB = input("OperandB")->template toType<DataType>();

            unsigned int size = result->shape().size();

            if constexpr ((DeviceUsed & (CPU_NAIVE)) !=0)
            {
                for(unsigned int e = 0; e<size; ++e)
                {
                    (*result)[e] = (*operandA)[e] + (*operandB)[e]*m_rate;
                }
            }
            else if constexpr ((DeviceUsed & (GPU_CUDA)) !=0)
            {
                elementwiseAddCUDAKernel<DataType>(operandA->gpuDataHandle(), operandB->gpuDataHandle(), m_rate, result->gpuDataHandle(), size);


            }
        }
    };
}
#endif
