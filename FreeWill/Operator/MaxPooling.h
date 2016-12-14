#ifndef MAXPOOLING_H
#define MAXPOOLING_H

#include "Operator.h"

namespace FreeWill
{
    template<DeviceType DeviceUsed = CPU, typename DataType = float>
    class MaxPooling : public Operator<DeviceUsed>
    {
    protected:
        using Operator<DeviceUsed>::input;
        using Operator<DeviceUsed>::output;

    public:
        MaxPooling()
            :Operator<DeviceUsed>({"Input"},{"Output", "SwitchX", "SwitchY"})
        {
        }

        virtual bool init() override
        {
            if (!input("Input") || !output("Output"))
            {
                return false;
            }

            if (input("Input")->shape().dimension() != 4)
            {
                return false;
            }

            if (input("Output")->shape().dimension() != 4)
            {
                return false;
            }

            if (output("SwitchX")->shape() != output("Output")->shape())
            {
                return false;
            }

            if (output("SwitchY")->shape() != output("Output")->shape())
            {
                return false;
            }

            if (input("Input")->shape()[0] != output("Output")->shape()[0])
            {
                return false;
            }

            if (input("Input")->shape()[1] != 2 * output("Output")->shape()[1])
            {
                return false;
            }

            if (input("Input")->shape()[2]!=2*output("Output")->shape()[2])
            {
                return false;
            }

            if (input("Input")->shape()[3]!=output("Output")->shape()[3])
            {
                return false;
            }

            return true;
        }

        virtual void evaluate() override
        {
            Tensor<DeviceUsed, DataType> *_input = (Tensor<DeviceUsed, DataType> *) input("Input");
            Tensor<DeviceUsed, DataType> *_output = (Tensor<DeviceUsed, DataType> *) output("Output");
            Tensor<DeviceUsed, DataType> *_switchX = (Tensor<DeviceUsed, DataType> *) output("SwitchX");
            Tensor<DeviceUsed, DataType> *_switchY = (Tensor<DeviceUsed, DataType> *) output("SwitchY");

            unsigned int newWidth = _output->shape()[1];
            unsigned int newHeight = _output->shape()[2];
            unsigned int oldWidth = _input->shape()[1];
            unsigned int oldHeight = _input->shape()[2];
            unsigned int batchSize = _output->shape()[3];

            unsigned int depthSize = _input->shape()[0];

            for (unsigned int b = 0; b < batchSize; ++b)
            {
                for(unsigned int y = 0;y< newHeight; ++y)
                {
                    for(unsigned int x =0;x<newWidth; ++x)
                    {

                        for(unsigned int depth = 0;depth<depthSize;++depth)
                        {

                            DataType a = _input[(b * oldWidth * oldHeight + y*2*oldWidth + x*2)*depthSize + depth];
                            DataType b = _input[(b * oldWidth * oldHeight + y*2*oldWidth + x*2 + 1)*depthSize + depth];
                            DataType c = _input[(b * oldWidth * oldHeight + (y*2+1)*oldWidth + x*2)*depthSize + depth];
                            DataType d = _input[(b * oldWidth * oldHeight + (y*2+1)*oldWidth + x*2 + 1)*depthSize +depth];
                        
                            DataType max = a;
                            unsigned int locationX = x*2;
                            unsigned int locationY = y*2;
                            if (b > max)
                            {
                                locationX = x*2+1;
                                locationY = y*2;
                                max = b;
                            }

                            if (c > max)
                            {
                                locationX =x* 2;
                                locationY = y*2+1;
                                max = c;
                            }

                            if (d > max)
                            {
                                locationX = x*2+1;
                                locationY = y*2+1;
                                max = d;
                            }

                            _output[(b * newWidth*newHeight + newWidth * y + x)*depthSize + depth] = max;
                            _switchX[(b* newWidth*newHeight + newWidth *y +x)*depthSize + depth] = locationX;
                            _switchY[(b* newWidth*newHeight + newWidth *y +x)*depthSize + depth] = locationY;
                        }
                    }
                }
            }
        }
    };
}

#endif
