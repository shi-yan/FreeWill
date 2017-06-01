#ifndef MAXPOOLINGDERIVATIVE_H
#define MAXPOOLINGDERIVATIVE_H

#include "Operator.h"
#include <cudnn.h>

namespace FreeWill
{

    template<DeviceType DeviceUsed = DeviceType::CPU_NAIVE, typename DataType = float>
    class MaxPoolingDerivative : public Operator<DeviceUsed>
    {
    protected:
        using Operator<DeviceUsed>::input;
        using Operator<DeviceUsed>::output;
        using Operator<DeviceUsed>::m_deviceId;

        cudnnPoolingDescriptor_t m_poolingDescriptor;
        cudnnTensorDescriptor_t m_outputGPUTensorDescriptor;
        cudnnTensorDescriptor_t m_outputDeltaGPUTensorDescriptor;
        cudnnTensorDescriptor_t m_inputGPUTensorDescriptor;
        cudnnTensorDescriptor_t m_inputDeltaGPUTensorDescriptor;


    public:
        MaxPoolingDerivative(unsigned int deviceId = 0)
            :Operator<DeviceUsed>({"Output","OutputGrad","Input", "SwitchX", "SwitchY"},{"InputGrad"}, deviceId),
            m_poolingDescriptor(0),
            m_outputGPUTensorDescriptor(0),
            m_outputDeltaGPUTensorDescriptor(0),
            m_inputGPUTensorDescriptor(0),
            m_inputDeltaGPUTensorDescriptor(0)
        {
            CHECK_GPU;

            if constexpr (DeviceUsed == DeviceType::GPU_CUDA)
            {
                RUN_CUDNN(cudnnCreatePoolingDescriptor(&m_poolingDescriptor));
                RUN_CUDNN(cudnnCreateTensorDescriptor(&m_outputGPUTensorDescriptor));
                RUN_CUDNN(cudnnCreateTensorDescriptor(&m_outputDeltaGPUTensorDescriptor));
                RUN_CUDNN(cudnnCreateTensorDescriptor(&m_inputGPUTensorDescriptor));
                RUN_CUDNN(cudnnCreateTensorDescriptor(&m_inputDeltaGPUTensorDescriptor));
            }
        }

        ~MaxPoolingDerivative()
        {
            CHECK_GPU;

            if constexpr (DeviceUsed == DeviceType::GPU_CUDA)
            {
                RUN_CUDNN(cudnnDestroyPoolingDescriptor(m_poolingDescriptor));
                RUN_CUDNN(cudnnDestroyTensorDescriptor(m_outputGPUTensorDescriptor));
                RUN_CUDNN(cudnnDestroyTensorDescriptor(m_outputDeltaGPUTensorDescriptor));
                RUN_CUDNN(cudnnDestroyTensorDescriptor(m_inputGPUTensorDescriptor));
                RUN_CUDNN(cudnnDestroyTensorDescriptor(m_inputDeltaGPUTensorDescriptor));

                m_poolingDescriptor = 0;
                m_outputGPUTensorDescriptor = 0;
                m_outputDeltaGPUTensorDescriptor = 0;
                m_inputGPUTensorDescriptor = 0;
                m_inputDeltaGPUTensorDescriptor = 0;
            }
        }

        virtual bool init() override
        {
            CHECK_GPU;

            FAIL_IF (!input("OutputGrad") || !output("InputGrad"));

            if constexpr (DeviceUsed == DeviceType::CPU_NAIVE)
            {
                FAIL_IF(!input("SwitchX") || !input("SwitchY"));
                FAIL_IF (input("OutputGrad")->shape() != input("SwitchX")->shape());
                FAIL_IF (input("OutputGrad")->shape() != input("SwitchY")->shape());
            }
            else if constexpr (DeviceUsed == DeviceType::GPU_CUDA)
            {
                FAIL_IF(!input("Output") || !input("Input"));
            }

            FAIL_IF (output("InputGrad")->shape()[0] != input("OutputGrad")->shape()[0]);

            FAIL_IF (output("InputGrad")->shape()[1] != input("OutputGrad")->shape()[1] * 2);

            FAIL_IF (output("InputGrad")->shape()[2] != input("OutputGrad")->shape()[2]*2);

            FAIL_IF (output("InputGrad")->shape()[3] != input("OutputGrad")->shape()[3]);
            
            if constexpr (DeviceUsed == DeviceType::GPU_CUDA)
            {
                cudnnDataType_t dataType = CUDNN_DATA_FLOAT;
                if constexpr (std::is_same<DataType,float>::value)
                {
                    dataType = CUDNN_DATA_FLOAT;
                }
                else if constexpr (std::is_same<DataType,double>::value)
                {
                    dataType = CUDNN_DATA_DOUBLE;
                }

                unsigned int batchSize = input("Input")->shape()[3];
                unsigned int channelSize = input("Input")->shape()[0];
                unsigned int width = input("Input")->shape()[1];
                unsigned int height = input("Input")->shape()[2];
       
                RUN_CUDNN(cudnnSetPooling2dDescriptor( m_poolingDescriptor,
                                                       CUDNN_POOLING_MAX,
                                                       CUDNN_NOT_PROPAGATE_NAN,
                                                       2,
                                                       2,
                                                       0,
                                                       0,
                                                       2,
                                                       2));

                RUN_CUDNN(cudnnSetTensor4dDescriptor( m_inputGPUTensorDescriptor,
                                                      CUDNN_TENSOR_NHWC,
                                                      dataType,
                                                      batchSize,
                                                      channelSize,
                                                      height, width));

                RUN_CUDNN(cudnnSetTensor4dDescriptor(m_outputGPUTensorDescriptor,
                                                     CUDNN_TENSOR_NHWC,
                                                     dataType,
                                                     batchSize,
                                                     channelSize,
                                                     height/2, width/2));
 
                RUN_CUDNN(cudnnSetTensor4dDescriptor( m_inputDeltaGPUTensorDescriptor,
                                                      CUDNN_TENSOR_NHWC,
                                                      dataType,
                                                      batchSize,
                                                      channelSize,
                                                      height, width));

                RUN_CUDNN(cudnnSetTensor4dDescriptor(m_outputDeltaGPUTensorDescriptor,
                                                     CUDNN_TENSOR_NHWC,
                                                     dataType,
                                                     batchSize,
                                                     channelSize,
                                                     height/2, width/2));
                         
            }

            return true;
        }

        virtual void evaluate() override
        {
            CHECK_GPU;

            Tensor<DeviceUsed, DataType> *_inputGrad = output("InputGrad")->template toType<DataType>();
            Tensor<DeviceUsed, DataType> *_outputGrad = input("OutputGrad")->template toType<DataType>();


            unsigned int outputWidth = _outputGrad->shape()[1];
            unsigned int outputHeight = _outputGrad->shape()[2];
            unsigned int batchSize = _outputGrad->shape()[3];
            unsigned int depthSize = _outputGrad->shape()[0];

            if constexpr (DeviceUsed == DeviceType::CPU_NAIVE)
            {
                Tensor<DeviceUsed, unsigned int> *_switchX = input("SwitchX")->template toType<unsigned int>();
                Tensor<DeviceUsed, unsigned int> *_switchY = input("SwitchY")->template toType<unsigned int>();


                for(unsigned int b = 0;b<batchSize;++b)
                {
                    for(unsigned int y = 0; y<outputHeight;++y)
                    {
                        for(unsigned int x = 0;x<outputWidth;++x)
                        {
                            for(unsigned int depth = 0;depth<depthSize;++depth)
                            {
                                unsigned int index = (b*outputWidth*outputHeight + y*outputWidth +x)*depthSize + depth;
                                unsigned int inputX = (*_switchX)[index];
                                unsigned int inputY = (*_switchY)[index];

                                (*_inputGrad)[(b*outputWidth*outputHeight*4 + inputY*outputWidth*2 + inputX)*depthSize + depth] = 
                                    (*_outputGrad)[index];
                            }
                        }
                    }
                }
            }
            else if constexpr (DeviceUsed == DeviceType::GPU_CUDA)
            {
                Tensor<DeviceUsed, DataType> *_input = input("Input")->template toType<DataType>();
                Tensor<DeviceUsed, DataType> *_output = input("Output")->template toType<DataType>();

                DataType alpha = 1.0;
                DataType beta = 0.0;

                RUN_CUDNN(cudnnPoolingBackward( Context<DeviceUsed>::getSingleton().cudnnHandle(m_deviceId),
                                                m_poolingDescriptor,
                                                &alpha,
                                                m_outputGPUTensorDescriptor,
                                                _output->gpuDataHandle(),
                                                m_outputDeltaGPUTensorDescriptor,
                                                _outputGrad->gpuDataHandle(),
                                                m_inputGPUTensorDescriptor,
                                                _input->gpuDataHandle(),
                                                &beta,
                                                m_inputDeltaGPUTensorDescriptor,
                                                _inputGrad->gpuDataHandle()));            
            }
        }
    };
}

#endif
