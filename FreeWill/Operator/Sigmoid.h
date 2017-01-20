#ifndef SIGMOID_H
#define SIGMOID_H

#include "Operator.h"
#include "../DeviceSelection.h"
#include <cmath>

#include <cuda.h>
#include <cudnn.h>
#include "../Context/Context.h"


namespace FreeWill
{

    template<DeviceType DeviceUsed = CPU, typename DataType = float>
    class Sigmoid : public Operator<DeviceUsed>
    {

    private:
        cudnnActivationDescriptor_t m_cudnnActivationDescriptor;

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

            if constexpr ((DeviceUsed & (GPU | GPU_CUDA)) != 0)
            {
                
                RUN_CUDNN(cudnnCreateActivationDescriptor(&m_cudnnActivationDescriptor));

                RUN_CUDNN(cudnnSetActivationDescriptor(m_cudnnActivationDescriptor, CUDNN_ACTIVATION_SIGMOID, CUDNN_PROPAGATE_NAN, 20.0));
            }

            return true;
        }

        void evaluate() override
        {

            Tensor<DeviceUsed, DataType> *inputTensor = (Tensor<DeviceUsed, DataType> *) m_inputParameters["Input"].m_tensors[0];
            Tensor<DeviceUsed, DataType> *outputTensor = (Tensor<DeviceUsed, DataType> *) m_outputParameters["Output"].m_tensors[0];


            if constexpr ((DeviceUsed & (CPU_SIMD | CPU_NAIVE)) != 0)
            {
                unsigned int size = m_inputParameters["Input"].m_tensors[0]->shape().size();

                for(unsigned int i = 0; i < size; ++i)
                {
                    (*outputTensor)[i] = 1 / (1 + exp(-(*inputTensor)[i]));
                }
            }
            else if constexpr ((DeviceUsed & (GPU | GPU_CUDA)) != 0)
            {
               DataType alpha = 1.0;
               DataType beta = 0.0;

/*               printf("%d\n", Context<DeviceUsed>::getSingleton().cudnnHandle());
               printf("%d\n", m_cudnnActivationDescriptor);
               printf("%d\n", inputTensor->gpuTensorDescriptor());
               printf("%d\n", inputTensor->gpuDataHandle());
               printf("%d\n", outputTensor->gpuTensorDescriptor());
               printf("%d\n", outputTensor->gpuDataHandle());
*/
               RUN_CUDNN(cudnnActivationForward(Context<DeviceUsed>::getSingleton().cudnnHandle(),
                                                m_cudnnActivationDescriptor,
                                                &alpha,
                                                inputTensor->gpuTensorDescriptor(),
                                                inputTensor->gpuDataHandle(),
                                                &beta,
                                                outputTensor->gpuTensorDescriptor(),
                                                outputTensor->gpuDataHandle())); 
            }
        }

        ~Sigmoid()
        {
            if constexpr ((DeviceUsed & (GPU | GPU_CUDA)) != 0)
            {
                RUN_CUDNN(cudnnDestroyActivationDescriptor(m_cudnnActivationDescriptor));
            }        
        }
     };

}
#endif
