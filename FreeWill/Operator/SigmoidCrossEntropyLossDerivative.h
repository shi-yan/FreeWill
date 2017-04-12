#ifndef SIGMOIDCROSSENTROPYLOSSDERIVATIVE_H
#define SIGMOIDCROSSENTROPYLOSSDERIVATIVE_H

#include "Operator.h"
#include "CrossEntropyLoss_CUDA.h"

namespace FreeWill
{
    template<DeviceType DeviceUsed = CPU, typename DataType = float>
    class SigmoidCrossEntropyLossDerivative : public Operator<DeviceUsed>
    {
    protected:
        using Operator<DeviceUsed>::input;
        using Operator<DeviceUsed>::output;

    public:
        SigmoidCrossEntropyLossDerivative()
        :Operator<DeviceUsed>({"Input", "Label"},{"Output"})
        {
        
        }

        virtual bool init() override
        {
            if (!input("Input") || !output("Output") || !input("Label"))
            {
                return false;
            }

            if ((input("Input")->shape().dimension() != 2 ) || 
                    (output("Output")->shape().dimension() != 2) || 
                    (input("Label")->shape().dimension() != 2))
            {
                return false;
            }

            if ((input("Input")->shape() != input("Label")->shape()) || (input("Input")->shape() != output("Output")->shape()))
            {
                return false;
            }

            return true;
        }

        virtual void evaluate() override
        {
            Tensor<DeviceUsed, DataType> *_input = input("Input")->template toType<DataType>();
            Tensor<DeviceUsed, DataType> *_output = output("Output")->template toType<DataType>();
            Tensor<DeviceUsed, DataType> *_label = input("Label")->template toType<DataType>();

            unsigned int batchSize = _input->shape()[1];
            unsigned int vectorSize = _input->shape()[0];

            //printf("batchSize %d vectorSize %d\n", batchSize, vectorSize);
            //printf("%d, %d, %d\n", _input->shape().size(), _output->shape().size(), _label->shape().size());
            
            if constexpr ((DeviceUsed & (CPU | CPU_NAIVE)) != 0)
            {
                for(unsigned int e = 0;e<batchSize;++e)
                {
                    for(unsigned int i = 0; i < vectorSize; ++i)
                    {
                        // DataType _inputSigmoid = 1.0 / (1.0 + exp(-(*_input)[e*vectorSize + i]));
                   
                        (*_output)[e * vectorSize + i] = (*_input)[e*vectorSize + i] - (*_label)[e * vectorSize + i];
                    }
                }
            }
            else if constexpr ((DeviceUsed & (GPU_CUDA | GPU)) != 0)
            {
                sigmoidCrossEntropyLossDerivativeCUDAKernel<DataType>(_input->gpuDataHandle(), _label->gpuDataHandle(), _output->gpuDataHandle(), vectorSize * batchSize);            
            }
        }

            
    };
}

#endif
