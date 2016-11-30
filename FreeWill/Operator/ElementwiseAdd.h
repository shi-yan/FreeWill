#ifndef ELEMENTWISEADD_H
#define ELEMENTWISEADD_H

#include "../DeviceSelection.h"
#include "Operator.h"
#include "../Tensor/Tensor.h"

namespace FreeWill
{
    template<int Dimension = 1, DeviceType DeviceUsed = CPU, typename DataType = float>
    class ElementwiseAdd : public Operator<DeviceUsed>
    {
        Tensor<Dimension, DeviceUsed, DataType> *m_operandA;
        Tensor<Dimension, DeviceUsed, DataType> *m_operandB;
        Tensor<Dimension, DeviceUsed, DataType> *m_result;
    public:
        ElementwiseAdd()
            :Operator<DeviceUsed>(),
            m_operandA(nullptr),
            m_operandB(nullptr),
            m_result(nullptr)
        {

        }

        void setOperandA(Tensor<Dimension, DeviceUsed, DataType> *operandA)
        {
            m_operandA = operandA;
        }

        void setOperandB(Tensor<Dimension, DeviceUsed, DataType> *operandB)
        {
            m_operandB = operandB;
        }

        void setResult(Tensor<Dimension, DeviceUsed, DataType> *result)
        {
            m_result = result;
        }

        virtual bool init() override
        {
            if (!m_operandA || !m_operandB || !m_result)
            {
                return false;
            }            

            if (m_operandA->shape() != m_operandB->shape() || m_operandA->shape() != m_result->shape())
            {
                return false;
            }

            return true;
        }

        virtual void evaluate() override
        {
            if constexpr (DeviceUsed == CPU_NAIVE)
            {
                unsigned int size = m_operandA->shape().size();
                for(unsigned int e = 0; e<size; ++e)
                {
                    (*m_result)[e] = (*m_operandA)[e] + (*m_operandB)[e];                    
                }
            }
            else
            {

            }
        }
    };
}
#endif
