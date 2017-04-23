#ifndef SIGMOIDCROSSENTROPYLOSS_H
#define SIGMOIDCROSSENTROPYLOSS_H

#include "Operator.h"
#include "../DeviceSelection.h"

#include "CrossEntropyLoss_CUDA.h"

namespace FreeWill
{
    template<DeviceType DeviceUsed = DeviceType::CPU_NAIVE, typename DataType = float>
    class CrossEntropyLoss : public Operator<DeviceUsed>
    {
    protected:
        using Operator<DeviceUsed>::input;
        using Operator<DeviceUsed>::output;

    public:
        CrossEntropyLoss()
            :Operator<DeviceUsed>({"Input", "Label"},{"Cost"})
        {}

        virtual bool init() override
        {
            FAIL_IF(!input("Input") || !output("Cost") || !input("Label"));
          
            FAIL_IF (input("Input")->shape().dimension() != 2 || output("Cost")->shape().dimension() != 2);

            FAIL_IF (input("Input")->shape() != input("Label")->shape());

            FAIL_IF (output("Cost")->shape()[0] != 1);

            FAIL_IF (input("Input")->shape()[1] != output("Cost")->shape()[1]);

            return true;
        }

        virtual void evaluate() override
        {
            Tensor<DeviceUsed, DataType> *_input = input("Input")->template toType<DataType>();
            Tensor<DeviceUsed, DataType> *_label = input("Label")->template toType<DataType>();

            Tensor<DeviceUsed, DataType> *_cost = output("Cost")->template toType<DataType>();

            unsigned int batchSize = _cost->shape()[1];
            unsigned int vectorSize = _input->shape()[0];

            if constexpr (DeviceUsed == DeviceType::CPU_NAIVE)
            {
                for(unsigned int e = 0; e< batchSize; ++e)
                {
                    (*_cost)[e] = 0;
                    for(size_t i = 0; i < vectorSize; ++i)
                    {
                        (*_cost)[e] += (*_label)[e * vectorSize + i]*log((*_input)[e * vectorSize + i]) 
                            + (1.0 - (*_label)[e*vectorSize +i])*log(1.0 - (*_input)[e* vectorSize + i]);
                    }
                
                    (*_cost)[e] *= -1.0;
                }
            }
            else if constexpr (DeviceUsed == DeviceType::GPU_CUDA)
            {
                if constexpr (std::is_same<float, DataType>::value)
                {
                    crossEntropyLossCUDAKernel<DataType>(_input->gpuDataHandle(), _label->gpuDataHandle(), _cost->gpuDataHandle(), vectorSize, batchSize);
                }
                else 
                {
                    //#if __CUDA_ARCH__ >= 600
                    //printf("cost size:%d\n", _cost->shape().size());
                    crossEntropyLossCUDAKernel<DataType>(_input->gpuDataHandle(), _label->gpuDataHandle(), _cost->gpuDataHandle(), vectorSize, batchSize);
                    //#else
                    //#warning "Cross Entropy CUDA kernel is not implemented yet for double type due to compute capability < 6.0"
                    //#endif
                }
            }
        }
    };
}

#endif
