#ifndef CONVOLUTION_H
#define CONVOLUTION_H


#include "Operator.h"

namespace FreeWill
{


    template<DeviceType DeviceUsed = CPU, typename DataType = float>
    class Convolution : public Operator<DeviceUsed>
    {
    protected:
        using Operator<DeviceUsed>::input;
        using Operator<DeviceUsed>::output;
        
        unsigned int m_zeroPaddingX;
        unsigned int m_strideX;
        unsigned int m_zeroPaddingY;
        unsigned int m_strideY;
        

    public:
        Convolution(unsigned int strideX = 1, unsigned int strideY = 1, 
                unsigned int zeroPaddingX = 0, unsigned int zeroPaddingY = 0)
            :Operator<DeviceUsed>({"Input", "FeatureMap", "Bias"}, {"Output"}),
            m_zeroPaddingX(zeroPaddingX),
            m_strideX(strideX),
            m_zeroPaddingY(zeroPaddingY),
            m_strideY(strideY)
        {}

        virtual bool init() override
        {
            if (!input("Input") || !input("FeatureMap") || !output("Output"))
            {
                return false;
            }

            if (input("Input")->shape().dimension() != 4)
            {
                return false;
            }

            if (input("FeatureMap")->shape().dimension() != 4)
            {
                return false;
            }

            if (output("Output")->shape().dimension() != 4)
            {
                return false;
            }

            if (input("Input")->shape()[0] != input("FeatureMap")->shape()[0])
            {
                return false;
            }

            if (input("FeatureMap")->shape()[1] != input("FeatureMap")->shape()[2])
            {
                return false;
            }

            unsigned int originalWidth = input("Input")->shape()[1];
            unsigned int originalHeight = input("Input")->shape()[2];

            unsigned int filterSize = input("FeatureMap")->shape()[1];

            if ((originalWidth - filterSize + 2*m_zeroPaddingX) % m_strideX != 0)
            {
                return false;
            }

            if ((originalHeight - filterSize + 2*m_zeroPaddingY) % m_strideY !=0)
            {
                return false;
            }

            unsigned int newWidth = (originalWidth - filterSize + 2*m_zeroPaddingX) / m_strideX;
            unsigned int newHeight = (originalHeight - filterSize + 2*m_zeroPaddingY) / m_strideY;

            if (output("Output")->shape()[1] != newWidth || output("Output")->shape()[2] != newHeight)
            {
                return false;
            }

            if (input("Bias")->shape().dimension() != 2)
            {
                return false;
            }

            if (input("Bias")->shape()[1] != input("FeatureMap")->shape()[3])
            {
                return false;
            }

            if (input("FeatureMap")->shape()[3] != output("Output")->shape()[0])
            {
                return false;
            }

            if (input("Input")->shape()[3] != output("Output")->shape()[3])
            {
                return false;
            }

            return true;
        }

        virtual void evaluate() override
        {
            Tensor<DeviceUsed, DataType> *_input = (Tensor<DeviceUsed, DataType> *) input("Input");
            Tensor<DeviceUsed, DataType> *_featureMap = (Tensor<DeviceUsed, DataType> *) input("FeatureMap");
            Tensor<DeviceUsed, DataType> *_bias = (Tensor<DeviceUsed, DataType> *) input("Bias");
            Tensor<DeviceUsed, DataType> *_output = (Tensor<DeviceUsed, DataType> *) output("Output");

            unsigned int featureMapCount = _featureMap->shape()[3];
            unsigned int featureMapLength = _featureMap->shape()[1];

            unsigned int originalWidth = _input->shape()[1];
            unsigned int originalHeight = _input->shape()[2];

            unsigned int channelCount = _featureMap->shape()[0];

            
            unsigned int newWidth = (originalWidth - featureMapLength + 2 * m_zeroPaddingX ) / m_strideX;
            unsigned int newHeight = (originalHeight - featureMapLength + 2 * m_zeroPaddingY) / m_strideY;

            unsigned int batchSize = _input->shape()[3];

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

                                    if ((realX >= 0 && realX < (int)originalWidth) && (y>=0 && y< (int)originalHeight))
                                    {
                                        unsigned int originalBaseIndex = (b* originalHeight * originalWidth + realY*originalWidth + realX)*channelCount;
                                
                                        for(unsigned int c = 0;c<channelCount;++c)
                                        {
                                            (*_output)[resultBaseIndex + k] += (*_featureMap)[(k * (featureMapLength * featureMapLength) + y*featureMapLength +x) * channelCount + c]
                                                * (*_input)[originalBaseIndex + c];
                                        }
                                    }
                                }
                            }

                            (*_output)[resultBaseIndex + k] += (*_bias)[b*featureMapCount + k]; 
                        }
                    }
                }
            }
        }
    };
}

#endif
