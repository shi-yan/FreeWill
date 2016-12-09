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
        
        unsigned int m_zeroPadding;
        unsigned int m_stride;
        
        

    public:
        Convolution(unsigned int stride = 1, unsigned int zeroPadding = 0)
            :Operator<DeviceUsed>({"Input", "FeatureMap", "Bias"}, {"Output"}),
            m_zeroPadding(zeroPadding),
            m_stride(stride)
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

            if (input("Output")->shape().dimension() != 4)
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

            unsigned int newWidth = (originalWidth - filterSize + 2*m_zeroPadding) / m_stride;
            unsigned int newHeight = (originalHeight - filterSize + 2*m_zeroPadding) / m_stride;

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

            return true;
        }

        virtual void evaluate() override
        {
        
        }
    };
}

#endif
