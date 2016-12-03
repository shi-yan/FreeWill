#ifndef SIGMOID_H
#define SIGMOID_H

#include "Operator.h"
#include "../DeviceSelection.h"
#include <cmath>

namespace FreeWill
{

    template<DeviceType DeviceUsed = CPU, typename DataType = float>
    class Sigmoid : public Operator<DeviceUsed>
    {
    protected:
        using Operator<DeviceUsed>::m_inputParameters;
        using Operator<DeviceUsed>::m_outputParameters;

    public:
        Sigmoid()
            :Operator<DeviceUsed>({"Input"}, {"Output"})
        {}  

        bool init() override
        {
            if (m_inputParameters["Input"].m_tensors.size() != 1 || m_outputParameters["Output"].m_tensors.size() != 1)
            {
                return false;
            }

            if (m_inputParameters["Input"].m_tensors[0]->shape() != m_outputParameters["Output"].m_tensors[0]->shape())
            {
                return false;
            }

            return true;
        }

        void evaluate() override
        {
            if constexpr ((DeviceUsed & (CPU_SIMD | CPU_NAIVE)) != 0)
            {
                unsigned int size = m_inputParameters["Input"].m_tensors[0]->shape().size();

                Tensor<DeviceUsed, DataType> *inputTensor = (Tensor<DeviceUsed, DataType> *) m_inputParameters["Input"].m_tensors[0];
                Tensor<DeviceUsed, DataType> *outputTensor = (Tensor<DeviceUsed, DataType> *) m_outputParameters["Output"].m_tensors[0];

                for(unsigned int i = 0; i < size; ++i)
                {
                    (*outputTensor)[i] = 1 / (1 + exp(-(*inputTensor)[i]));
                }
            }
        }
     };

}
#endif
