#ifndef MAXPOOLINGDERIVATIVE_H
#define MAXPOOLINGDERIVATIVE_H

#include "Operator.h"


namespace FreeWill
{

    template<DeviceType DeviceUsed = CPU, typename DataType = float>
    class MaxPoolingDerivative : public Operator<DeviceUsed>
    {
    protected:
        using Operator<DeviceUsed>::input;
        using Operator<DeviceUsed>::output;

    public:
        MaxPoolingDerivative()
            :Operator<DeviceUsed>({"OutputGrad", "SwitchX", "SwitchY"},{"InputGrad"})
        {
        }

        virtual bool init() override
        {
            if (!input("OutputGrad") || !input("SwitchX") || !input("SwitchY") || !output("InputGrad"))
            {
                return false;
            }

            if (input("OutputGrad")->shape() != input("SwitchX")->shape())
            {
                return false;
            }

            if (input("OutputGrad")->shape() != input("SwitchY")->shape())
            {
                return false;
            }

            if (output("InputGrad")->shape()[0] != input("OutputGrad")->shape()[0])
            {
                return false;
            }

            if (output("InputGrad")->shape()[1] != input("OutputGrad")->shape()[1] * 2)
            {
                return false;
            }

            if (output("InputGrad")->shape()[2] != input("OutputGrad")->shape()[2]*2)
            {
                return false;
            }

            if (output("InputGrad")->shape()[3] != input("OutputGrad")->shape()[3])
            {
                return false;
            }

            return true;
        }

        virtual void evaluate() override
        {
            Tensor<DeviceUsed, DataType> *_inputGrad = (Tensor<DeviceUsed, DataType> *) output("InputGrad");
            Tensor<DeviceUsed, DataType> *_outputGrad = (Tensor<DeviceUsed, DataType> *) input("OutputGrad");
            Tensor<DeviceUsed, unsigned int> *_switchX = (Tensor<DeviceUsed, unsigned int>*) input("SwitchX");
            Tensor<DeviceUsed, unsigned int> *_switchY = (Tensor<DeviceUsed, unsigned int> *) input("SwitchY");


            unsigned int outputWidth = _outputGrad->shape()[1];
            unsigned int outputHeight = _outputGrad->shape()[2];
            unsigned int batchSize = _outputGrad->shape()[3];
            unsigned int depthSize = _outputGrad->shape()[0];

            for(unsigned int b = 0;b<batchSize;++b)
            {
                for(unsigned int y = 0; y<outputHeight;++y)
                {
                    for(unsigned int x = 0;x<outputWidth;++x)
                    {
                        for(unsigned int depth = 0;depth<depthSize;++depth)
                        {
                            unsigned int index = (b*outputWidth*outputHeight + y*outputWidth +x)*depthSize + depth;
                            unsigned int inputX = _switchX[index];
                            unsigned int inputY = _switchY[index];

                            _inputGrad[(b*outputWidth*outputHeight*4 + inputY*outputWidth*2 + inputX)*depthSize + depth] = 
                                _outputGrad[index];
                        }
                    }
                }
            }
        }
    };
}

#endif
