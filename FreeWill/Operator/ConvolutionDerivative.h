#ifndef CONVOLUTIONDERIVATIVE_H
#define CONVOLUTIONDERIVATIVE_H

#include "Operator.h"
#include "../Context/Context.h"

namespace FreeWill
{

    template<DeviceType DeviceUsed = CPU, typename DataType = float>
    class ConvolutionDerivative : public Operator<DeviceUsed>
    {
    protected:
        using Operator<DeviceUsed>::input;
        using Operator<DeviceUsed>::output;
        unsigned int m_strideX;
        unsigned int m_strideY;
        unsigned int m_zeroPaddingX;
        unsigned int m_zeroPaddingY;

        cudnnTensorDescriptor_t m_prevActivationGPUTensorDescriptor;
        cudnnTensorDescriptor_t m_outputDeltaGPUTensorDescriptor;
        cudnnTensorDescriptor_t m_biasGradGPUTensorDescriptor;
        cudnnFilterDescriptor_t m_featureMapFilterDescriptor;
        cudnnConvolutionDescriptor_t m_convolutionDescriptor;
        cudnnConvolutionBwdFilterAlgo_t m_filterBackwardAlgorithm;

    public:
        ConvolutionDerivative(unsigned int strideX = 1, unsigned int strideY = 1,
                unsigned int zeroPaddingX = 0, unsigned int zeroPaddingY = 0)
            :Operator<DeviceUsed>({"PrevActivation","OutputGrad","FeatureMap"},{"FeatureMapGrad","BiasGrad","InputGrad"}),
            m_strideX(strideX),
            m_strideY(strideY),
            m_zeroPaddingX(zeroPaddingX),
            m_zeroPaddingY(zeroPaddingY),
            m_prevActivationGPUTensorDescriptor(0),
            m_outputDeltaGPUTensorDescriptor(0),
            m_biasGradGPUTensorDescriptor(0),
            m_featureMapFilterDescriptor(0),
            m_convolutionDescriptor(0),
            m_filterBackwardAlgorithm()
        {
            if ((DeviceUsed & (GPU | GPU_CUDA)) != 0)
            {
                RUN_CUDNN(cudnnCreateTensorDescriptor(&m_prevActivationGPUTensorDescriptor));
                RUN_CUDNN(cudnnCreateTensorDescriptor(&m_outputDeltaGPUTensorDescriptor));
                RUN_CUDNN(cudnnCreateTensorDescriptor(&m_biasGradGPUTensorDescriptor));
                RUN_CUDNN(cudnnCreateFilterDescriptor(&m_featureMapFilterDescriptor));
                RUN_CUDNN(cudnnCreateConvolutionDescriptor(&m_convolutionDescriptor));
            }
        }

        ~ConvolutionDerivative()
        {
            if ((DeviceUsed & (GPU | GPU_CUDA)) != 0)
            {
                RUN_CUDNN(cudnnDestroyTensorDescriptor(m_prevActivationGPUTensorDescriptor));
                RUN_CUDNN(cudnnDestroyTensorDescriptor(m_outputDeltaGPUTensorDescriptor));
                RUN_CUDNN(cudnnDestroyTensorDescriptor(m_biasGradGPUTensorDescriptor));
                RUN_CUDNN(cudnnDestroyFilterDescriptor(m_featureMapFilterDescriptor));
                RUN_CUDNN(cudnnDestroyConvolutionDescriptor(m_convolutionDescriptor));

                m_prevActivationGPUTensorDescriptor = 0;
                m_outputDeltaGPUTensorDescriptor = 0;
                m_biasGradGPUTensorDescriptor = 0;
                m_featureMapFilterDescriptor = 0;
                m_convolutionDescriptor = 0;
            }
        }

        void displayFilterBackwardAlgorithm(cudnnConvolutionBwdFilterAlgo_t algorithm)
        {
            QString message = "Convolution filter bacward algorithm:";
            switch (algorithm)
            {
            ENUM_CASE(CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0, message)
            ENUM_CASE(CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1, message)
            ENUM_CASE(CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT, message)
            ENUM_CASE(CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3, message)
            ENUM_CASE(CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED, message)
            default:
                qDebug() << "Warning: unrecognized convolution filter backward algorithm:" << algorithm;
                break;
            }
        }

        virtual bool init() override
        {
            FAIL_IF (!input("PrevActivation") || !input("OutputGrad") || !input("FeatureMap"));

            FAIL_IF (!output("FeatureMapGrad") || !output("BiasGrad") || !output("InputGrad"));

            FAIL_IF (input("PrevActivation")->shape()[0] != input("FeatureMap")->shape()[0]);

            FAIL_IF (input("PrevActivation")->shape().dimension() != 4);

            FAIL_IF (input("FeatureMap")->shape().dimension() != 4);

            FAIL_IF (input("OutputGrad")->shape().dimension() != 4);

            FAIL_IF (input("FeatureMap")->shape()[1] != input("FeatureMap")->shape()[2]);

            FAIL_IF (output("InputGrad")->shape() != input("PrevActivation")->shape());

            unsigned int originalWidth = input("PrevActivation")->shape()[1];
            unsigned int originalHeight = input("PrevActivation")->shape()[2];
            unsigned int filterSize = input("FeatureMap")->shape()[1];

            FAIL_IF ((originalWidth - filterSize + 2*m_zeroPaddingX) % m_strideX != 0);

            FAIL_IF ((originalHeight - filterSize + 2*m_zeroPaddingY) % m_strideY !=0);

            unsigned int newWidth = (originalWidth - filterSize + 2*m_zeroPaddingX) / m_strideX + 1;
            unsigned int newHeight = (originalHeight - filterSize + 2*m_zeroPaddingY) / m_strideY + 1;

            //qDebug() << "output" << output("Output")->shape()[1] <<";"<< output("Output")->shape()[2];

            FAIL_IF (input("OutputGrad")->shape()[1] != newWidth || input("OutputGrad")->shape()[2] != newHeight);

            FAIL_IF (output("BiasGrad")->shape().dimension() != 1);

            FAIL_IF (output("BiasGrad")->shape()[0] != input("FeatureMap")->shape()[3]);

            FAIL_IF (input("FeatureMap")->shape()[3] != input("OutputGrad")->shape()[0]);

            FAIL_IF (input("PrevActivation")->shape()[3] != input("OutputGrad")->shape()[3]);

            FAIL_IF (input("FeatureMap")->shape() != output("FeatureMapGrad")->shape());

            unsigned int channelCount = input("PrevActivation")->shape()[0];
            unsigned int batchSize = input("PrevActivation")->shape()[3];

            unsigned int filterCount = input("FeatureMap")->shape()[3];



            if constexpr ((DeviceUsed & (GPU | GPU_CUDA)) != 0)
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

                RUN_CUDNN(cudnnSetTensor4dDescriptor( m_prevActivationGPUTensorDescriptor,
                                                      CUDNN_TENSOR_NHWC,
                                                      dataType,
                                                      batchSize,
                                                      channelCount,
                                                      originalHeight,
                                                      originalWidth));

                RUN_CUDNN(cudnnSetTensor4dDescriptor( m_outputDeltaGPUTensorDescriptor,
                                                      CUDNN_TENSOR_NHWC,
                                                      dataType,
                                                      batchSize,
                                                      filterCount,
                                                      newHeight,
                                                      newWidth));


                RUN_CUDNN(cudnnSetTensor4dDescriptor( m_biasGradGPUTensorDescriptor,
                                                      CUDNN_TENSOR_NHWC,
                                                      dataType,
                                                      1,
                                                      filterCount,
                                                      1,
                                                      1));

                //qDebug() << "filterCount" << filterCount << "channelCount" << channelCount;

                RUN_CUDNN(cudnnSetFilter4dDescriptor( m_featureMapFilterDescriptor,
                                                      dataType,
                                                      CUDNN_TENSOR_NHWC,
                                                      filterCount,
                                                      channelCount,
                                                      filterSize,
                                                      filterSize));

                //qDebug() <<"zero padding stride:" << m_zeroPaddingX << m_zeroPaddingY << m_strideX << m_strideY;
                RUN_CUDNN(cudnnSetConvolution2dDescriptor( m_convolutionDescriptor,
                                                           m_zeroPaddingY ,
                                                           m_zeroPaddingX ,
                                                           m_strideY,
                                                           m_strideX,
                                                           1,
                                                           1,
                                                           CUDNN_CROSS_CORRELATION ));


                RUN_CUDNN(cudnnGetConvolutionBackwardFilterAlgorithm( Context<DeviceUsed>::getSingleton().cudnnHandle(),
                                                                      m_prevActivationGPUTensorDescriptor,
                                                                      m_outputDeltaGPUTensorDescriptor,
                                                                      m_convolutionDescriptor,
                                                                      m_featureMapFilterDescriptor,
                                                                      CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
                                                                      0,
                                                                      &m_filterBackwardAlgorithm ));

                displayFilterBackwardAlgorithm(m_filterBackwardAlgorithm);

                const int requestedAlgoCount = 6;
                cudnnConvolutionBwdFilterAlgoPerf_t perfResults[requestedAlgoCount];
                int returnedAlgoCount = 0;

                RUN_CUDNN(cudnnFindConvolutionBackwardFilterAlgorithm( Context<DeviceUsed>::getSingleton().cudnnHandle(),
                                                                       m_prevActivationGPUTensorDescriptor,
                                                                       m_outputDeltaGPUTensorDescriptor,
                                                                       m_convolutionDescriptor,
                                                                       m_featureMapFilterDescriptor,
                                                                       requestedAlgoCount,
                                                                       &returnedAlgoCount,
                                                                       perfResults ));

                qDebug() << returnedAlgoCount << "convolution filter backward algorithm benchmarks:";

                for(int i =0;i<returnedAlgoCount;++i)
                {
                    qDebug() << i << "Status:" << perfResults[i].status << "Time:" << perfResults[i].time << "milliseconds" << "Memory need:" << perfResults[i].memory;

                    displayFilterBackwardAlgorithm(perfResults[i].algo);
                }
            }

            return true;
        }

        virtual void evaluate() override
        {
            Tensor<DeviceUsed, DataType> *_prevActivation = (Tensor<DeviceUsed, DataType> *) input("PrevActivation");
            Tensor<DeviceUsed, DataType> *_featureMap = (Tensor<DeviceUsed, DataType> *) input("FeatureMap");
            Tensor<DeviceUsed, DataType> *_outputGrad = (Tensor<DeviceUsed, DataType> *) input("OutputGrad");

            Tensor<DeviceUsed, DataType> *_featureMapGrad = (Tensor<DeviceUsed, DataType> *) output("FeatureMapGrad");
            Tensor<DeviceUsed, DataType> *_biasGrad = (Tensor<DeviceUsed, DataType> *) output("BiasGrad");
            Tensor<DeviceUsed, DataType> *_inputGrad = (Tensor<DeviceUsed, DataType> *) output("InputGrad");

            unsigned int featureMapCount = _featureMap->shape()[3];
            unsigned int featureMapLength = _featureMap->shape()[1];

            unsigned int originalWidth = _prevActivation->shape()[1];
            unsigned int originalHeight = _prevActivation->shape()[2];

            unsigned int channelCount = _featureMap->shape()[0];

            
            unsigned int newWidth = (originalWidth - featureMapLength + 2 * m_zeroPaddingX ) / m_strideX + 1;
            unsigned int newHeight = (originalHeight - featureMapLength + 2 * m_zeroPaddingY) / m_strideY + 1;

            unsigned int batchSize = _prevActivation->shape()[3];
            if constexpr ((DeviceUsed & (CPU | CPU_NAIVE)) != 0)
            {
                for (unsigned int b = 0; b < batchSize; ++b)
                {
                    for(unsigned int newIndexY = 0; newIndexY < newHeight;++newIndexY)
                    {
                        for (unsigned int newIndexX = 0; newIndexX < newWidth;++newIndexX)
                        {

                            int startX = -m_zeroPaddingX + newIndexX * m_strideX;
                            int startY = -m_zeroPaddingY + newIndexY * m_strideY;

                            unsigned int resultBaseIndex = (b * newWidth*newHeight +newIndexY * newWidth + newIndexX) * featureMapCount; 

                            for (unsigned int k = 0; k < featureMapCount; ++k)
                            {

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
                                            unsigned int featureMapBaseIndex = (k*(featureMapLength * featureMapLength) + y*featureMapLength + x) * channelCount; 
                                
                                            for(unsigned int c = 0;c<channelCount;++c)
                                            {
                                                /*(*_output)[resultBaseIndex + k] += 
                                                    (*_featureMap)[(k * (featureMapLength * featureMapLength) + 
                                                            y*featureMapLength +x) * channelCount + c]
                                                    * (*_input)[originalBaseIndex + c];
                                                */

                                                (*_featureMapGrad)[featureMapBaseIndex + c]
                                                    += (*_outputGrad)[resultBaseIndex + k] * (*_prevActivation)[originalBaseIndex + c];

                                                (*_inputGrad)[originalBaseIndex + c] += (*_featureMap)[(k * (featureMapLength * featureMapLength) + 
                                                    y*featureMapLength + x)*channelCount + c] * (*_outputGrad)[resultBaseIndex + k];
                                             
                                            }
                                        
                                            //qDebug() << "base index" << realX << ";" << realY << ";" <<originalWidth <<";"<< originalBaseIndex;
                                            //qDebug() << "feature map" << (*_featureMap)[(k * (featureMapLength * featureMapLength) +
                                            //            y*featureMapLength +x) * channelCount + 0];                                   
                                            //qDebug() << (*_input)[originalBaseIndex ];
                                        }
                                    }
                                }

                                //qDebug() << "result loc" << resultBaseIndex + k;
                                (*_biasGrad)[k] += (*_outputGrad)[resultBaseIndex + k];
                                //(*_output)[resultBaseIndex + k] += (*_bias)[k]; 
                            }
                        }
                    }                
                }

                DataType scale = 1.0 / (newWidth * newHeight);


               /* for(unsigned int k = 0;k<_biasGrad->shape().size();++k)
                {
                    (*_biasGrad)[k] *= scale;
                }*/

/*                for(unsigned int i = 0;i< _featureMapGrad->shape().size();++i)
                {
                    (*_featureMapGrad)[i] *= scale;
                }
*/
                for(unsigned int i =0;i<_inputGrad->shape().size(); ++i)
                {
                    (*_inputGrad)[i] *= scale;
                }

            }
            else if constexpr ((DeviceUsed & (GPU | GPU_CUDA)) !=0 )
            {
                DataType alpha = 1.0;
                DataType beta = 0.0;

                RUN_CUDNN(cudnnConvolutionBackwardFilter(Context<DeviceUsed>::getSingleton().cudnnHandle(),
                                                        &alpha,
                                                        m_prevActivationGPUTensorDescriptor,
                                                        _prevActivation->gpuDataHandle(),
                                                        m_outputDeltaGPUTensorDescriptor,
                                                        _outputGrad->gpuDataHandle(),
                                                        m_convolutionDescriptor,
                                                        m_filterBackwardAlgorithm,
                                                        nullptr,
                                                        0,
                                                        &beta,
                                                        m_featureMapFilterDescriptor,
                                                        _featureMapGrad->gpuDataHandle()));

                RUN_CUDNN(cudnnConvolutionBackwardBias(Context<DeviceUsed>::getSingleton().cudnnHandle(),
                                                       &alpha,
                                                       m_outputDeltaGPUTensorDescriptor,
                                                       _outputGrad->gpuDataHandle(),
                                                       &beta,
                                                       m_biasGradGPUTensorDescriptor,
                                                       _biasGrad->gpuDataHandle()));
            }
            

        }

    };
}

#endif
