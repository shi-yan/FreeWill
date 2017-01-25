#ifndef ACTIVATIONDERIVATIVE_H
#define ACTIVATIONDERIVATIVE_H

#include "Operator.h"
#include <cuda.h>
#include <cudnn.h>
#include "../Context/Context.h"
#include "Activation.h"

namespace FreeWill
{
    template <ActivationMode ActivationModeUsed = SIGMOID, DeviceType DeviceUsed = CPU, typename DataType = float>
    class ActivationDerivative : public Operator<DeviceUsed>
    {

    private:
        cudnnActivationDescriptor_t m_cudnnActivationDescriptor;

    protected:
        using Operator<DeviceUsed>::input;
        using Operator<DeviceUsed>::output;

    public:        
        ActivationDerivative()
            :Operator<DeviceUsed>({"Input","Output","OutputDelta"},{"InputDelta"}),
            m_cudnnActivationDescriptor(0)
        {}

        ~ActivationDerivative()
        {
            clear();
        }

        virtual bool init() override
        {
            if (!input("Output") || !output("InputDelta") || !input("OutputDelta"))
            {
                return false;
            }

            if (input("Output")->shape() != output("InputDelta")->shape())
            {
                return false;
            }

            if (input("OutputDelta")->shape() != input("Output")->shape())
            {
                return false;
            }
            

            if constexpr ((DeviceUsed & (GPU | GPU_CUDA)) != 0)
            {

                if (!m_cudnnActivationDescriptor)
                {
                    RUN_CUDNN(cudnnCreateActivationDescriptor(&m_cudnnActivationDescriptor));
                }

                if constexpr (ActivationModeUsed == SIGMOID)
                {
                    RUN_CUDNN(cudnnSetActivationDescriptor(m_cudnnActivationDescriptor, CUDNN_ACTIVATION_SIGMOID, CUDNN_PROPAGATE_NAN, 20.0));
                
                }
                else if constexpr (ActivationModeUsed == RELU)
                {
                    RUN_CUDNN(cudnnSetActivationDescriptor(m_cudnnActivationDescriptor, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 20.0));
                
                }
                else if constexpr (ActivationModeUsed == TANH)
                {
                    RUN_CUDNN(cudnnSetActivationDescriptor(m_cudnnActivationDescriptor, CUDNN_ACTIVATION_TANH, CUDNN_PROPAGATE_NAN, 20.0));
               
                }
                else if constexpr (ActivationModeUsed == CLIPPED_RELU)
                {
                    RUN_CUDNN(cudnnSetActivationDescriptor(m_cudnnActivationDescriptor, CUDNN_ACTIVATION_CLIPPED_RELU, CUDNN_PROPAGATE_NAN, 20.0));
   
                }
            }

            return true;
        }


        virtual void evaluate() override
        {
            unsigned int size = input("Output")->shape().size();

            Tensor<DeviceUsed, DataType> *_output = (Tensor<DeviceUsed, DataType> *) input("Output");
            Tensor<DeviceUsed, DataType> *_inputDelta = (Tensor<DeviceUsed, DataType> *) output("InputDelta");
            Tensor<DeviceUsed, DataType> *_outputDelta = (Tensor<DeviceUsed, DataType> *) input("OutputDelta");

            if constexpr ((DeviceUsed & (CPU | CPU_NAIVE | CPU_SIMD)) != 0)
            {
                if constexpr (ActivationModeUsed == SIGMOID)
                {
                    for (unsigned int i =0; i<size; ++i)
                    {
                        (*_inputDelta)[i] = (*_output)[i] * (1.0 - (*_output)[i]) * (*_outputDelta)[i];
                    }
                }
                else if constexpr (ActivationModeUsed == RELU) 
                {
                    for(unsigned int i =0;i<size; ++i)
                    {
                        (*_inputDelta)[i] = ((*_output)[i] > 0.0 ? (*_output)[i] : 0.0) * (*_outputDelta)[i];
                    }
                }
                else if constexpr (ActivationModeUsed == TANH)
                {
                }
                else if constexpr (ActivationModeUsed == CLIPPED_RELU)
                {
           
                }
            }
            else if constexpr ((DeviceUsed & (GPU | GPU_CUDA)) != 0)
            {
                DataType alpha = 1.0;
                DataType beta = 0.0;
                RUN_CUDNN(cudnnActivationBackward(Context<DeviceUsed>::getSingleton().cudnnHandle(),
                                                m_cudnnActivationDescriptor,
                                                &alpha,
                                                _output->gpuTensorDescriptor(),
                                                _output->gpuDataHandle(),
                                                _outputDelta->gpuTensorDescriptor(),
                                                _outputDelta->gpuDataHandle(),
                                                _output->gpuTensorDescriptor(),
                                                _output->gpuDataHandle(),
                                                &beta,
                                                _inputDelta->gpuTensorDescriptor(),
                                                _inputDelta->gpuDataHandle()
                                                
                                                )); 
            
            }
        }

        void clear() override
        {
            Operator<DeviceUsed>::clear();

            if constexpr ((DeviceUsed & (GPU | GPU_CUDA)) != 0)
            {
                RUN_CUDNN(cudnnDestroyActivationDescriptor(m_cudnnActivationDescriptor));
                m_cudnnActivationDescriptor = 0;
            }        
        }
 
    };
}

#endif
