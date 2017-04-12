#ifndef SOFTMAXLOGLOSSDERIVATIVE_H
#define SOFTMAXLOGLOSSDERIVATIVE_H

#include "Operator.h"
#include "SoftmaxLogLoss_CUDA.h"

namespace FreeWill
{

    template<DeviceType DeviceUsed = CPU, typename DataType = float>
    class SoftmaxLogLossDerivative : public Operator<DeviceUsed>
    {
    protected:
        using Operator<DeviceUsed>::input;
        using Operator<DeviceUsed>::output;

    public:
        SoftmaxLogLossDerivative() : Operator<DeviceUsed>({"Output", "Label"},{"InputGrad"})
        {
        }

        virtual bool init() override
        {
            FAIL_IF (!input("Output") || !input("Label") || !output("InputGrad"));

            FAIL_IF (input("Output")->shape() != output("InputGrad")->shape());

            FAIL_IF (input("Output")->shape().dimension() != 2);

            FAIL_IF (input("Label")->shape().dimension() != 1);

            FAIL_IF (input("Output")->shape()[1] != input("Label")->shape()[0]);

            return true;
        }

        virtual void evaluate() override
        {
            Tensor<DeviceUsed, DataType> *_output = input("Output")->template toType<DataType>();
            Tensor<DeviceUsed, unsigned int> *_label = input("Label")->template toType<unsigned int>();
            Tensor<DeviceUsed, DataType> *_inputGrad = output("InputGrad")->template toType<DataType>();

            unsigned int batchSize = _output->shape()[1];
            unsigned int vectorSize = _output->shape()[0];

            if constexpr ((DeviceUsed & (CPU | CPU_NAIVE)) != 0)
            {
                for(unsigned int b = 0;b<batchSize;++b)
                {

                    for(unsigned int i = 0;i<vectorSize;++i)
                    {
                        (*_inputGrad)[b*vectorSize+ i] = (*_output)[b*vectorSize +i];
                    }

                    (*_inputGrad)[b*vectorSize + (*_label)[b]] -= 1.0;
                }
            }
            else if constexpr ((DeviceUsed & (GPU | GPU_CUDA)) != 0)
            {
                softmaxLogLossDerivativeCUDAKernel<DataType>(_inputGrad->gpuDataHandle(), _output->gpuDataHandle(), _label->gpuDataHandle(), vectorSize, batchSize);
            }
        }
    };
}

#endif
