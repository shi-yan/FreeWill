#ifndef ACTIVATIONDERIVATIVE_H
#define ACTIVATIONDERIVATIVE_H

#include "Operator.h"
#include <cuda.h>
#include <cudnn.h>
#include "../Context/Context.h"
#include "Activation.h"

namespace FreeWill
{
    template <ActivationMode ActivationModeUsed = ActivationMode::SIGMOID, DeviceType DeviceUsed = DeviceType::CPU_NAIVE, typename DataType = float>
    class ActivationDerivative : public Operator<DeviceUsed>
    {

    private:
        cudnnActivationDescriptor_t m_cudnnActivationDescriptor;

    protected:
        using Operator<DeviceUsed>::input;
        using Operator<DeviceUsed>::output;
        using Operator<DeviceUsed>::m_deviceId;

    public:        
        ActivationDerivative(unsigned int deviceId = 0)
            :Operator<DeviceUsed>({"Input","Output","OutputDelta"},{"InputDelta"},deviceId),
            m_cudnnActivationDescriptor(0)
        {}

        ~ActivationDerivative()
        {
            clear();
        }

        virtual bool init() override
        {
            CHECK_GPU;

            FAIL_IF (!input("Output") || !output("InputDelta") || !input("OutputDelta"));

            FAIL_IF (input("Output")->shape() != output("InputDelta")->shape());

            FAIL_IF (input("OutputDelta")->shape() != input("Output")->shape());
            

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


        virtual void evaluate() override
        {
            CHECK_GPU;
            unsigned int size = input("Output")->shape().size();

            Tensor<DeviceUsed, DataType> *_output = input("Output")->template toType<DataType>();
            Tensor<DeviceUsed, DataType> *_inputDelta = output("InputDelta")->template toType<DataType>();
            Tensor<DeviceUsed, DataType> *_outputDelta = input("OutputDelta")->template toType<DataType>();

            if constexpr (DeviceUsed == DeviceType::CPU_NAIVE)
            {
                if constexpr (ActivationModeUsed == ActivationMode::SIGMOID)
                {
                    for (unsigned int i =0; i<size; ++i)
                    {
                        (*_inputDelta)[i] = (*_output)[i] * (1.0 - (*_output)[i]) * (*_outputDelta)[i];
                    }
                }
                else if constexpr (ActivationModeUsed == ActivationMode::RELU)
                {
                    for(unsigned int i =0;i<size; ++i)
                    {
                        (*_inputDelta)[i] = ((*_output)[i] > 0.0 ? 1.0 : 0.0) * (*_outputDelta)[i];
                    }
                }
                else if constexpr (ActivationModeUsed == ActivationMode::TANH)
                {
                    (void) _output;
                    (void) _inputDelta;
                    (void) _outputDelta;
                }
                else if constexpr (ActivationModeUsed == ActivationMode::CLIPPED_RELU)
                {
                    (void) _output;
                    (void) _inputDelta;
                    (void) _outputDelta;
                }
            }
            else if constexpr (DeviceUsed == DeviceType::GPU_CUDA)
            {
                DataType alpha = 1.0;
                DataType beta = 0.0;
                RUN_CUDNN(cudnnActivationBackward(Context<DeviceUsed>::getSingleton().cudnnHandle(m_deviceId),
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

            if constexpr (DeviceUsed == DeviceType::GPU_CUDA)
            {
                RUN_CUDNN(cudnnDestroyActivationDescriptor(m_cudnnActivationDescriptor));
                m_cudnnActivationDescriptor = 0;
            }        
        }
 
    };
}

#endif
