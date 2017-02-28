#ifndef ELEMENTWISEADD_H
#define ELEMENTWISEADD_H

#include "../DeviceSelection.h"
#include "Operator.h"
#include "../Tensor/Tensor.h"

#include "ElementwiseAdd_CUDA.h"

namespace FreeWill
{
    template<DeviceType DeviceUsed = CPU, typename DataType = float>
    class ElementwiseAdd : public Operator<DeviceUsed>
    {
        DataType m_rate;
   public:
        ElementwiseAdd(DataType rate = 1.0f)
            :Operator<DeviceUsed>({"Operand"}, {"Result"}),
            m_rate(rate)
        {
        }

        void setRate(DataType rate)
        {
            m_rate = rate;
        }

        virtual bool init() override
        {
            FAIL_IF (Operator<DeviceUsed>::m_inputParameters["Operand"].m_tensors.size() < 1);

            FAIL_IF (Operator<DeviceUsed>::m_outputParameters["Result"].m_tensors.size() != 1);

            for(unsigned int i = 0; i< Operator<DeviceUsed>::m_inputParameters["Operand"].m_tensors.size(); ++i)
            {
               FAIL_IF (Operator<DeviceUsed>::m_inputParameters["Operand"].m_tensors[i]->shape().size()
                       != Operator<DeviceUsed>::m_outputParameters["Result"].m_tensors[0]->shape().size());
            }

            return true;
        }

        virtual void evaluate() override
        {

            Tensor<DeviceUsed, DataType> *result = (Tensor<DeviceUsed, DataType> *) Operator<DeviceUsed>::m_outputParameters["Result"].m_tensors[0];
            Tensor<DeviceUsed, DataType> *operandA = (Tensor<DeviceUsed, DataType> *) Operator<DeviceUsed>::m_inputParameters["Operand"].m_tensors[0];
            Tensor<DeviceUsed, DataType> *operandB = (Tensor<DeviceUsed, DataType> *) Operator<DeviceUsed>::m_inputParameters["Operand"].m_tensors[1];

            unsigned int size = result->shape().size();

            if constexpr (DeviceUsed == CPU_NAIVE)
            {
                for(unsigned int e = 0; e<size; ++e)
                {
                    (*result)[e] = (*operandA)[e] + (*operandB)[e]*m_rate;
                }
            }
            else if constexpr ((DeviceUsed & (GPU | GPU_CUDA)) !=0)
            {
                elementwiseAddCUDAKernel<DataType>(operandA->gpuDataHandle(), operandB->gpuDataHandle(), m_rate, result->gpuDataHandle(), size);


            }
        }
    };
}
#endif
