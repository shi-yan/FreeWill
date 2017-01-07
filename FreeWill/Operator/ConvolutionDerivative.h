#ifndef CONVOLUTIONDERIVATIVE_H
#define CONVOLUTIONDERIVATIVE_H

#include "Operator.h"

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

    public:
        ConvolutionDerivative(unsigned int strideX = 1, unsigned int strideY = 1,
                unsigned int zeroPaddingX = 0, unsigned int zeroPaddingY = 0)
            :Operator<DeviceUsed>({"PrevActivation","OutputGrad","FeatureMap"},{"FeatureMapGrad","BiasGrad","InputGrad"}),
            m_strideX(strideX),
            m_strideY(strideY),
            m_zeroPaddingX(zeroPaddingX),
            m_zeroPaddingY(zeroPaddingY)
        {
        }

        virtual bool init() override
        {
            if (!input("PrevActivation") || !input("OutputGrad") || !input("FeatureMap"))
            {
                return false;
            }

            if (!output("FeatureMapGrad") || !output("BiasGrad") || !output("InputGrad"))
            {
                return false;
            }

            if (input("PrevActivation")->shape()[0] != input("FeatureMap")->shape()[0])
            {
                return false;
            }

            if (input("PrevActivation")->shape().dimension() != 4)
            {
                return false;
            }

            if (input("FeatureMap")->shape().dimension() != 4)
            {
                return false;
            }

            if (input("OutputGrad")->shape().dimension() != 4)
            {
                return false;
            }

            if (input("FeatureMap")->shape()[1] != input("FeatureMap")->shape()[2])
            {
                return false;
            }

            if (output("InputGrad")->shape() != input("PrevActivation")->shape())
            {
                return false;
            }

            unsigned int originalWidth = input("PrevActivation")->shape()[1];
            unsigned int originalHeight = input("PrevActivation")->shape()[2];

            unsigned int filterSize = input("FeatureMap")->shape()[1];

            if ((originalWidth - filterSize + 2*m_zeroPaddingX) % m_strideX != 0)
            {
                return false;
            }

            if ((originalHeight - filterSize + 2*m_zeroPaddingY) % m_strideY !=0)
            {
                return false;
            }

            unsigned int newWidth = (originalWidth - filterSize + 2*m_zeroPaddingX) / m_strideX + 1;
            unsigned int newHeight = (originalHeight - filterSize + 2*m_zeroPaddingY) / m_strideY + 1;

            //qDebug() << "output" << output("Output")->shape()[1] <<";"<< output("Output")->shape()[2];

            if (input("OutputGrad")->shape()[1] != newWidth || input("OutputGrad")->shape()[2] != newHeight)
            {
                return false;
            }

            if (output("BiasGrad")->shape().dimension() != 1)
            {
                return false;
            }

            if (output("BiasGrad")->shape()[0] != input("FeatureMap")->shape()[3])
            {
                return false;
            }

            if (input("FeatureMap")->shape()[3] != input("OutputGrad")->shape()[0])
            {
                return false;
            }

            if (input("PrevActivation")->shape()[3] != input("OutputGrad")->shape()[3])
            {
                return false;
            }

            if (input("FeatureMap")->shape() != output("FeatureMapGrad")->shape())
            {
                return false;
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


            for(unsigned int k = 0;k<_biasGrad->shape().size();++k)
            {
                (*_biasGrad)[k] *= scale;
            }

            for(unsigned int i = 0;i< _featureMapGrad->shape().size();++i)
            {
                (*_featureMapGrad)[i] *= scale;
            }

            for(unsigned int i =0;i<_inputGrad->shape().size(); ++i)
            {
                (*_inputGrad)[i] *= scale;
            }


        }

    };
}

#endif
