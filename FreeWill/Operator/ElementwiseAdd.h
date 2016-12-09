#ifndef ELEMENTWISEADD_H
#define ELEMENTWISEADD_H

#include "../DeviceSelection.h"
#include "Operator.h"
#include "../Tensor/Tensor.h"

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
            if (Operator<DeviceUsed>::m_inputParameters["Operand"].m_tensors.size() < 1)
            {
                return false;
            }

            if (Operator<DeviceUsed>::m_outputParameters["Result"].m_tensors.size() != 1)
            {
                return false;
            }

            for(unsigned int i = 0; i< Operator<DeviceUsed>::m_inputParameters["Operand"].m_tensors.size(); ++i)
            {
               if(Operator<DeviceUsed>::m_inputParameters["Operand"].m_tensors[i]->shape().size() 
                       != Operator<DeviceUsed>::m_outputParameters["Result"].m_tensors[0]->shape().size())
               {
                return false;
               }
            }

            return true;
        }

        virtual void evaluate() override
        {
            if constexpr (DeviceUsed == CPU_NAIVE)
            {

                Tensor<DeviceUsed, DataType> *result = (Tensor<DeviceUsed, DataType> *) Operator<DeviceUsed>::m_outputParameters["Result"].m_tensors[0];
                Tensor<DeviceUsed, DataType> *operandA = (Tensor<DeviceUsed, DataType> *) Operator<DeviceUsed>::m_inputParameters["Operand"].m_tensors[0];
                Tensor<DeviceUsed, DataType> *operandB = (Tensor<DeviceUsed, DataType> *) Operator<DeviceUsed>::m_inputParameters["Operand"].m_tensors[1];

                unsigned int size = result->shape().size();

                //printf("elementwise add size: %d\n", size);

                for(unsigned int e = 0; e<size; ++e)
                {
                    (*result)[e] = (*operandA)[e] + (*operandB)[e]*m_rate;
                }
            }
            else
            {

            }
        }
    };
}
#endif
