#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include <QDebug>
#include "Operator.h"
#include "../Context/Context.h"

namespace FreeWill
{


    template<DeviceType DeviceUsed = DeviceType::CPU_NAIVE, typename DataType = float>
    class Convolution : public Operator<DeviceUsed>
    {
    protected:
        using Operator<DeviceUsed>::input;
        using Operator<DeviceUsed>::output;
        using Operator<DeviceUsed>::m_deviceId;
        
        unsigned int m_zeroPaddingX;
        unsigned int m_strideX;
        unsigned int m_zeroPaddingY;
        unsigned int m_strideY;

        cudnnTensorDescriptor_t m_inputGPUTensorDescriptor;
        cudnnTensorDescriptor_t m_outputGPUTensorDescriptor;
        cudnnTensorDescriptor_t m_biasGPUTensorDescriptor;
        cudnnFilterDescriptor_t m_filterDescriptor;
        cudnnConvolutionDescriptor_t m_convolutionDescriptor;
        cudnnConvolutionFwdAlgo_t m_convolutionForwardAlgorithm;
        size_t m_workspaceSize;
        unsigned char *m_workspace;

    public:
        Convolution(unsigned int strideX = 1, unsigned int strideY = 1, 
                unsigned int zeroPaddingX = 0, unsigned int zeroPaddingY = 0, unsigned int deviceId = 0)
            :Operator<DeviceUsed>({"Input", "FeatureMap", "Bias"}, {"Output"}, deviceId),
            m_zeroPaddingX(zeroPaddingX),
            m_strideX(strideX),
            m_zeroPaddingY(zeroPaddingY),
            m_strideY(strideY),
            m_inputGPUTensorDescriptor(0),
            m_outputGPUTensorDescriptor(0),
            m_biasGPUTensorDescriptor(0),
            m_filterDescriptor(0),
            m_convolutionDescriptor(0),
            m_convolutionForwardAlgorithm(),
            m_workspaceSize(0),
            m_workspace(nullptr)
        {
            CHECK_GPU;
            if constexpr (DeviceUsed == DeviceType::GPU_CUDA)
            {
                RUN_CUDNN(cudnnCreateTensorDescriptor(&m_inputGPUTensorDescriptor));
                RUN_CUDNN(cudnnCreateTensorDescriptor(&m_outputGPUTensorDescriptor));
                RUN_CUDNN(cudnnCreateTensorDescriptor(&m_biasGPUTensorDescriptor));
                RUN_CUDNN(cudnnCreateFilterDescriptor(&m_filterDescriptor));
                RUN_CUDNN(cudnnCreateConvolutionDescriptor(&m_convolutionDescriptor));
            }
        }

        ~Convolution()
        {
            CHECK_GPU;
            if constexpr (DeviceUsed == DeviceType::GPU_CUDA)
            {
                RUN_CUDNN(cudnnDestroyTensorDescriptor(m_inputGPUTensorDescriptor));
                RUN_CUDNN(cudnnDestroyTensorDescriptor(m_outputGPUTensorDescriptor));
                RUN_CUDNN(cudnnDestroyTensorDescriptor(m_biasGPUTensorDescriptor));
                RUN_CUDNN(cudnnDestroyFilterDescriptor(m_filterDescriptor));
                RUN_CUDNN(cudnnDestroyConvolutionDescriptor(m_convolutionDescriptor));
                m_inputGPUTensorDescriptor = 0;
                m_outputGPUTensorDescriptor = 0;
                m_biasGPUTensorDescriptor = 0;
                m_filterDescriptor = 0;
                m_convolutionDescriptor = 0;
                m_biasGPUTensorDescriptor = 0;

                if (m_workspace)
                {
                    RUN_CUDA(cudaFree(m_workspace));
                    m_workspace = nullptr;
                }
            }
        }

        void displayConvolutionAlgorithm(cudnnConvolutionFwdAlgo_t algorithm)
        {
            QString message = "Convolution forward algorithm:";
            switch (algorithm)
            {
            ENUM_CASE(CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, message)
            ENUM_CASE(CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, message)
            ENUM_CASE(CUDNN_CONVOLUTION_FWD_ALGO_GEMM, message)
            ENUM_CASE(CUDNN_CONVOLUTION_FWD_ALGO_DIRECT, message)
            ENUM_CASE(CUDNN_CONVOLUTION_FWD_ALGO_FFT, message)
            ENUM_CASE(CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING, message)
            ENUM_CASE(CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD, message)
            ENUM_CASE(CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED, message)
            default:
                qDebug() << "Warning: unrecognized convolution forward algorithm:" << algorithm;
                break;
            }
        }

        void displayTensorDescriptor(cudnnTensorDescriptor_t descriptor)
        {
            int dimnb = 0;
            int dimA[4] = {0};
            int strideA[4] = {0};
            cudnnDataType_t dataType;
            RUN_CUDNN(cudnnGetTensorNdDescriptor(descriptor,
                                                 4,
                                                 &dataType,
                                                 &dimnb,
                                                 dimA,
                                                 strideA));

            printf("Tensor descriptor: %d, dim: %d,%d,%d,%d | stride: %d,%d,%d,%d\n", dimnb, dimA[0], dimA[1], dimA[2], dimA[3],strideA[0],strideA[1],strideA[2],strideA[3]);
        }

        static void reg()
        {
            OperatorRegistry<Convolution<DeviceUsed, DataType>>::m_operatorFactoryInitializer.getA();
        }

        virtual bool init() override
        {
            CHECK_GPU;

            FAIL_IF (!input("Input") || !input("FeatureMap") || !input("Bias") || !output("Output"));

            FAIL_IF (input("Input")->shape().dimension() != 4);

            FAIL_IF (input("FeatureMap")->shape().dimension() != 4);

            FAIL_IF (output("Output")->shape().dimension() != 4);

            FAIL_IF (input("Input")->shape()[0] != input("FeatureMap")->shape()[0]);

            FAIL_IF (input("FeatureMap")->shape()[1] != input("FeatureMap")->shape()[2]);

            unsigned int originalWidth = input("Input")->shape()[1];
            unsigned int originalHeight = input("Input")->shape()[2];
            unsigned int filterSize = input("FeatureMap")->shape()[1];

            unsigned int newWidth = (originalWidth - filterSize + 2*m_zeroPaddingX) / m_strideX + 1;
            unsigned int newHeight = (originalHeight - filterSize + 2*m_zeroPaddingY) / m_strideY + 1;

            FAIL_IF ((originalWidth - filterSize + 2*m_zeroPaddingX) % m_strideX != 0);
            FAIL_IF ((originalHeight - filterSize + 2*m_zeroPaddingY) % m_strideY !=0);

            //qDebug() << "output" << output("Output")->shape()[1] <<";"<< output("Output")->shape()[2];
            //qDebug() << "originalWidth" << originalWidth << "originalHeight" << originalHeight;
            //qDebug() << "newwidth" << newWidth << "newHeight" << newHeight;

            FAIL_IF (output("Output")->shape()[1] != newWidth || output("Output")->shape()[2] != newHeight);

            FAIL_IF (input("Bias")->shape().dimension() != 1);

            FAIL_IF (input("Bias")->shape()[0] != input("FeatureMap")->shape()[3]);

            FAIL_IF (input("FeatureMap")->shape()[3] != output("Output")->shape()[0]);

            FAIL_IF (input("Input")->shape()[3] != output("Output")->shape()[3]);

            if constexpr (DeviceUsed == DeviceType::GPU_CUDA)
            {
                unsigned int batchSize = input("Input")->shape()[3];
                unsigned int channelCount = input("FeatureMap")->shape()[0];
                unsigned int filterCount = input("FeatureMap")->shape()[3];

                cudnnDataType_t dataType = CUDNN_DATA_FLOAT;
                if constexpr (std::is_same<DataType,float>::value)
                {
                    dataType = CUDNN_DATA_FLOAT;
                }
                else if constexpr (std::is_same<DataType,double>::value)
                {
                    dataType = CUDNN_DATA_DOUBLE;
                }

                RUN_CUDNN(cudnnSetTensor4dDescriptor( m_inputGPUTensorDescriptor,
                                                      CUDNN_TENSOR_NHWC,
                                                      dataType,
                                                      batchSize,
                                                      channelCount,
                                                      originalHeight,
                                                      originalWidth));

                RUN_CUDNN(cudnnSetTensor4dDescriptor( m_outputGPUTensorDescriptor,
                                                      CUDNN_TENSOR_NHWC,
                                                      dataType,
                                                      batchSize,
                                                      filterCount,
                                                      newHeight,
                                                      newWidth));


                RUN_CUDNN(cudnnSetTensor4dDescriptor( m_biasGPUTensorDescriptor,
                                                      CUDNN_TENSOR_NHWC,
                                                      dataType,
                                                      1,
                                                      filterCount,
                                                      1,
                                                      1));

                //qDebug() << "filterCount" << filterCount << "channelCount" << channelCount;

                RUN_CUDNN(cudnnSetFilter4dDescriptor( m_filterDescriptor,
                                                      dataType,
                                                      CUDNN_TENSOR_NHWC,
                                                      filterCount,
                                                      channelCount,
                                                      filterSize,
                                                      filterSize));

                cudnnDataType_t cudnnDataType = cudnnDataType_t::CUDNN_DATA_FLOAT;

                if constexpr (std::is_same<DataType, float>::value)
                {
                   cudnnDataType = cudnnDataType_t::CUDNN_DATA_FLOAT;
                }
                else if constexpr (std::is_same<DataType, double>::value)
                {
                   cudnnDataType = cudnnDataType_t::CUDNN_DATA_DOUBLE;
                }

                //qDebug() <<"zero padding stride:" << m_zeroPaddingX << m_zeroPaddingY << m_strideX << m_strideY;
                RUN_CUDNN(cudnnSetConvolution2dDescriptor( m_convolutionDescriptor,
                                                           m_zeroPaddingY ,
                                                           m_zeroPaddingX ,
                                                           m_strideY,
                                                           m_strideX,
                                                           1,
                                                           1,
                                                           CUDNN_CROSS_CORRELATION, cudnnDataType));

                RUN_CUDNN(cudnnGetConvolutionForwardAlgorithm( Context<DeviceUsed>::getSingleton().cudnnHandle(m_deviceId),
                                                               m_inputGPUTensorDescriptor,
                                                               m_filterDescriptor,
                                                               m_convolutionDescriptor,
                                                               m_outputGPUTensorDescriptor,
                                                               CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                               0,
                                                               &m_convolutionForwardAlgorithm));

                qDebug() << "Convolution forward algorithm find based on huristic:";
                displayConvolutionAlgorithm(m_convolutionForwardAlgorithm);

                RUN_CUDNN(cudnnGetConvolutionForwardWorkspaceSize( Context<DeviceUsed>::getSingleton().cudnnHandle(m_deviceId),
                                                                          m_inputGPUTensorDescriptor,
                                                                          m_filterDescriptor,
                                                                          m_convolutionDescriptor,
                                                                          m_outputGPUTensorDescriptor,
                                                                          m_convolutionForwardAlgorithm,
                                                                          &m_workspaceSize));
                qDebug() << "Required workspace size:" << m_workspaceSize;


                int returnedAlgoCount = 0;
                const int requestedAlgoCount = 20;
                cudnnConvolutionFwdAlgoPerf_t perfResults[requestedAlgoCount];

                RUN_CUDNN(cudnnFindConvolutionForwardAlgorithm(  Context<DeviceUsed>::getSingleton().cudnnHandle(m_deviceId),
                                                                 m_inputGPUTensorDescriptor,
                                                                 m_filterDescriptor,
                                                                 m_convolutionDescriptor,
                                                                 m_outputGPUTensorDescriptor,
                                                                 requestedAlgoCount,
                                                                 &returnedAlgoCount,
                                                                 perfResults));

                qDebug() << returnedAlgoCount << "convolution forward algorithm benchmarks:";

                for(int i =0;i<returnedAlgoCount;++i)
                {
                    qDebug() << i << "Status:" << perfResults[i].status << "Time:" << perfResults[i].time << "milliseconds" << "Memory need:" << perfResults[i].memory;

                    displayConvolutionAlgorithm(perfResults[i].algo);
                }

                qDebug() << "----------------------------------------------------------------------------";

                if (m_convolutionForwardAlgorithm != perfResults[0].algo)
                {
                    m_convolutionForwardAlgorithm = perfResults[0].algo;
                    m_workspaceSize = perfResults[0].memory;
                }

                if (m_workspaceSize)
                {
                    RUN_CUDA(cudaMalloc(&m_workspace, m_workspaceSize));
                }
            }

            return true;
        }

        virtual void evaluate() override
        {
            CHECK_GPU;

            Tensor<DeviceUsed, DataType> *_input = input("Input")->template toType<DataType>();
            Tensor<DeviceUsed, DataType> *_featureMap = input("FeatureMap")->template toType<DataType>();
            Tensor<DeviceUsed, DataType> *_bias = input("Bias")->template toType<DataType>();
            Tensor<DeviceUsed, DataType> *_output = output("Output")->template toType<DataType>();

            unsigned int featureMapCount = _featureMap->shape()[3];
            unsigned int featureMapLength = _featureMap->shape()[1];

            unsigned int originalWidth = _input->shape()[1];
            unsigned int originalHeight = _input->shape()[2];

            unsigned int channelCount = _featureMap->shape()[0];

            
            unsigned int newWidth = (originalWidth - featureMapLength + 2 * m_zeroPaddingX ) / m_strideX + 1;
            unsigned int newHeight = (originalHeight - featureMapLength + 2 * m_zeroPaddingY) / m_strideY + 1;

            unsigned int batchSize = _input->shape()[3];

            if constexpr (DeviceUsed == DeviceType::CPU_NAIVE)
            {
                for (unsigned int b = 0; b < batchSize; ++b)
                {

                    for(unsigned int newIndexY = 0; newIndexY < newHeight;++newIndexY)
                    {
                        for (unsigned int newIndexX = 0; newIndexX < newWidth;++newIndexX)
                        {

                            int startX = -m_zeroPaddingX + newIndexX * m_strideX;
                            int startY = -m_zeroPaddingY + newIndexY * m_strideY;

                            for (unsigned int k = 0; k < featureMapCount; ++k)
                            {
                                unsigned int resultBaseIndex = (b * newWidth*newHeight +newIndexY * newWidth + newIndexX) * featureMapCount;


                                for(int y = 0; y< (int)featureMapLength; ++y)
                                {
                                    for(int x = 0; x < (int)featureMapLength; ++x)
                                    {
                                        int realX = x + startX;
                                        int realY = y + startY;

                                        if ((realX >= 0 && realX < (int)originalWidth)
                                            && (realY>=0 && realY< (int)originalHeight))
                                        {
                                            unsigned int originalBaseIndex = (b* originalHeight * originalWidth + realY*originalWidth + realX)
                                                *channelCount;
                                
                                            for(unsigned int c = 0;c<channelCount;++c)
                                            {
                                                (*_output)[resultBaseIndex + k] +=
                                                    (*_featureMap)[(k * (featureMapLength * featureMapLength) +
                                                        y*featureMapLength +x) * channelCount + c]
                                                    * (*_input)[originalBaseIndex + c];
                                            }
                                        
                                            //qDebug() << "base index" << realX << ";" << realY << ";" <<originalWidth <<";"<< originalBaseIndex;
                                            //qDebug() << "feature map" << (*_featureMap)[(k * (featureMapLength * featureMapLength) +
                                            //            y*featureMapLength +x) * channelCount + 0];
                                            //qDebug() << (*_input)[originalBaseIndex ];
                                        }
                                    }
                                }

                                //qDebug() << "result loc" << resultBaseIndex + k;

                                (*_output)[resultBaseIndex + k] += (*_bias)[k];
                            }
                        }
                    }
                }

            }
            else if constexpr (DeviceUsed == DeviceType::GPU_CUDA)
            {
                if (m_workspaceSize != 0)
                {
                    qDebug() << "Convolution forward algorithm requires workspace!";
                }

                float alpha = 1.0;
                float beta = 0.0;
                RUN_CUDNN(cudnnConvolutionForward( Context<DeviceUsed>::getSingleton().cudnnHandle(m_deviceId),
                                                   &alpha,
                                                   m_inputGPUTensorDescriptor,
                                                   _input->gpuDataHandle(),
                                                   m_filterDescriptor,
                                                   _featureMap->gpuDataHandle(),
                                                   m_convolutionDescriptor,
                                                   m_convolutionForwardAlgorithm,
                                                   m_workspace,
                                                   m_workspaceSize,
                                                   &beta,
                                                   m_outputGPUTensorDescriptor,
                                                   _output->gpuDataHandle()));

                beta = 1.0;

                    //displayTensorDescriptor(m_biasGPUTensorDescriptor);
                    //displayTensorDescriptor(m_inputGPUTensorDescriptor);
                    //displayTensorDescriptor(m_outputGPUTensorDescriptor);

                    //printf("bias size:%d\n", _bias->shape().size());
                    //printf("output size:%d\n", _output->shape().size());

                RUN_CUDNN(cudnnAddTensor( Context<DeviceUsed>::getSingleton().cudnnHandle(m_deviceId),
                                          &alpha,
                                          m_biasGPUTensorDescriptor,
                                          _bias->gpuDataHandle(),
                                          &beta,
                                          m_outputGPUTensorDescriptor,
                                          _output->gpuDataHandle()));


            }

        }
    };
}

#endif
