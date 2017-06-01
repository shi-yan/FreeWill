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
    enum class ActivationMode : uint32_t
    {
        SIGMOID,
        RELU,
        TANH,
        CLIPPED_RELU
    };

    template<ActivationMode ActivationModeUsed = ActivationMode::SIGMOID, DeviceType DeviceUsed = DeviceType::CPU_NAIVE, typename DataType = float>
    class Activation : public Operator<DeviceUsed>
    {

    private:
        cudnnActivationDescriptor_t m_cudnnActivationDescriptor;

    protected:
        using Operator<DeviceUsed>::input;
        using Operator<DeviceUsed>::output;
        using Operator<DeviceUsed>::m_deviceId;

    public:
        Activation(unsigned int deviceId = 0)
            :Operator<DeviceUsed>({"Input"}, {"Output"}, deviceId),
            m_cudnnActivationDescriptor(0)
        {}

        bool init() override
        {
            CHECK_GPU;

            FAIL_IF (!input("Input") || !output("Output"));

            FAIL_IF (input("Input")->shape() != output("Output")->shape());

            if constexpr (DeviceUsed == DeviceType::GPU_CUDA)
            {
                if (!m_cudnnActivationDescriptor)
                {
                    RUN_CUDNN(cudnnCreateActivationDescriptor(&m_cudnnActivationDescriptor));
                }

                if constexpr (ActivationModeUsed == ActivationMode::SIGMOID)
                {
                    RUN_CUDNN(cudnnSetActivationDescriptor(m_cudnnActivationDescriptor, CUDNN_ACTIVATION_SIGMOID, CUDNN_PROPAGATE_NAN, 20.0));
                }
                else if constexpr (ActivationModeUsed == ActivationMode::RELU)
                {
                    RUN_CUDNN(cudnnSetActivationDescriptor(m_cudnnActivationDescriptor, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 20.0));
                }
                else if constexpr (ActivationModeUsed == ActivationMode::TANH)
                {
                    RUN_CUDNN(cudnnSetActivationDescriptor(m_cudnnActivationDescriptor, CUDNN_ACTIVATION_TANH, CUDNN_PROPAGATE_NAN, 20.0));
                }
                else if constexpr (ActivationModeUsed == ActivationMode::CLIPPED_RELU)
                {
                    RUN_CUDNN(cudnnSetActivationDescriptor(m_cudnnActivationDescriptor, CUDNN_ACTIVATION_CLIPPED_RELU, CUDNN_PROPAGATE_NAN, 20.0));
                }
            }

            return true;
        }

        void evaluate() override
        {
            CHECK_GPU;

            Tensor<DeviceUsed, DataType> *_input = input("Input")->template toType<DataType>();
            Tensor<DeviceUsed, DataType> *_output = output("Output")->template toType<DataType>();


            if constexpr (DeviceUsed == DeviceType::CPU_NAIVE)
            {
                unsigned int size = _input->shape().size();

                if constexpr (ActivationModeUsed == ActivationMode::SIGMOID)
                {
                    for(unsigned int i = 0; i < size; ++i)
                    {
                        (*_output)[i] = 1 / (1 + exp(-(*_input)[i]));
                    }
                }
                else if constexpr (ActivationModeUsed == ActivationMode::RELU)
                {
                    for(unsigned int i =0;i<size; ++i)
                    {
                        (*_output)[i] = (*_input)[i] > 0.0 ? (*_input)[i] : 0.0;
                    }
                }
                else if constexpr (ActivationModeUsed == ActivationMode::TANH)
                {
                    (void) _input;
                    (void) _output;
                }
                else if constexpr (ActivationModeUsed == ActivationMode::CLIPPED_RELU)
                {
                    (void) _input;
                    (void) _output;
                }
            }
            else if constexpr (DeviceUsed == DeviceType::GPU_CUDA)
            {
               DataType alpha = 1.0;
               DataType beta = 0.0;

               RUN_CUDNN(cudnnActivationForward(Context<DeviceUsed>::getSingleton().cudnnHandle(m_deviceId),
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

            if constexpr (DeviceUsed == DeviceType::GPU_CUDA)
            {
                RUN_CUDNN(cudnnDestroyActivationDescriptor(m_cudnnActivationDescriptor));
                m_cudnnActivationDescriptor = 0;
            }        
            
        }
    };
}
#endif
