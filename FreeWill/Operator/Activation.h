#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "Operator.h"
#include "../DeviceSelection.h"
#include <cmath>

#include <cuda.h>
#include <cudnn.h>
#include "../Context/Context.h"


namespace FreeWill
{
    typedef enum
    {
        SIGMOID,
        RELU,
        TANH,
        CLIPPED_RELU
    } ActivationMode;

    template<ActivationMode ActivationModeUsed = SIGMOID, DeviceType DeviceUsed = CPU_NAIVE, typename DataType = float>
    class Activation : public Operator<DeviceUsed>
    {

    private:
        cudnnActivationDescriptor_t m_cudnnActivationDescriptor;

    protected:
        using Operator<DeviceUsed>::input;
        using Operator<DeviceUsed>::output;

    public:
        Activation()
            :Operator<DeviceUsed>({"Input"}, {"Output"}),
            m_cudnnActivationDescriptor(0)
        {}

        bool init() override
        {
            FAIL_IF (!input("Input") || !output("Output"));

            FAIL_IF (input("Input")->shape() != output("Output")->shape());

            if constexpr ((DeviceUsed & (GPU_CUDA)) != 0)
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

        void evaluate() override
        {

            Tensor<DeviceUsed, DataType> *_input = input("Input")->template toType<DataType>();
            Tensor<DeviceUsed, DataType> *_output = output("Output")->template toType<DataType>();


            if constexpr ((DeviceUsed & (CPU_NAIVE)) != 0)
            {
                unsigned int size = _input->shape().size();

                if constexpr (ActivationModeUsed == SIGMOID)
                {
                    for(unsigned int i = 0; i < size; ++i)
                    {
                        (*_output)[i] = 1 / (1 + exp(-(*_input)[i]));
                    }
                }
                else if constexpr (ActivationModeUsed == RELU)
                {
                    for(unsigned int i =0;i<size; ++i)
                    {
                        (*_output)[i] = (*_input)[i] > 0.0 ? (*_input)[i] : 0.0;
                    }
                }
                else if constexpr (ActivationModeUsed == TANH)
                {
                    (void) _input;
                    (void) _output;
                }
                else if constexpr (ActivationModeUsed == CLIPPED_RELU)
                {
                    (void) _input;
                    (void) _output;
                }
            }
            else if constexpr ((DeviceUsed & (GPU_CUDA)) != 0)
            {
               DataType alpha = 1.0;
               DataType beta = 0.0;

               RUN_CUDNN(cudnnActivationForward(Context<DeviceUsed>::getSingleton().cudnnHandle(),
                                                m_cudnnActivationDescriptor,
                                                &alpha,
                                                _input->gpuTensorDescriptor(),
                                                _input->gpuDataHandle(),
                                                &beta,
                                                _output->gpuTensorDescriptor(),
                                                _output->gpuDataHandle())); 
            }
        }

        ~Activation()
        {
            clear();
        }

        void clear() override
        {
            Operator<DeviceUsed>::clear();

            if constexpr ((DeviceUsed & (GPU_CUDA)) != 0)
            {
                RUN_CUDNN(cudnnDestroyActivationDescriptor(m_cudnnActivationDescriptor));
                m_cudnnActivationDescriptor = 0;
            }        
            
        }
    };
}
#endif
