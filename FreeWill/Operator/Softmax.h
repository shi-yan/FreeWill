#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "Operator.h"
#include "cublas_v2.h"

namespace FreeWill
{

    template<DeviceType DeviceUsed = CPU, typename DataType = float>
    class Softmax : public Operator<DeviceUsed>
    {
    protected:
        using Operator<DeviceUsed>::input;
        using Operator<DeviceUsed>::output;

    public:
        Softmax() : Operator<DeviceUsed>({"Input", "Label"},{"Cost","Output"})
        {
        }

        virtual bool init() override 
        {
            FAIL_IF (!input("Input") || !input("Label") || !output("Cost") || !output("Output"));

            FAIL_IF (input("Input")->shape() != output("Output")->shape());

            FAIL_IF (input("Input")->shape().dimension() != 2);

            FAIL_IF (input("Label")->shape().dimension() !=1 || output("Cost")->shape().dimension() != 1);

            unsigned int batchSize = input("Input")->shape()[1];

            FAIL_IF (batchSize != input("Label")->shape()[0] || batchSize != output("Cost")->shape()[0]);

            return true;
        }

        virtual void evaluate() override
        {
            Tensor<DeviceUsed, DataType> *_input = (Tensor<DeviceUsed, DataType> *) input("Input");
            Tensor<DeviceUsed, unsigned int> *_label = (Tensor<DeviceUsed, unsigned int> *) input("Label");
            Tensor<DeviceUsed, DataType> *_cost = (Tensor<DeviceUsed, DataType> *) output("Cost");
            Tensor<DeviceUsed, DataType> *_output = (Tensor<DeviceUsed, DataType> *) output("Output");

            unsigned int batchSize = _input->shape()[1];

            for(unsigned int b = 0; b < batchSize; ++b)
            {
                unsigned int vectorSize = _input->shape()[0];

                DataType maximum = (*_input)[b * vectorSize];

                for(unsigned int i = 1;i<vectorSize;++i)
                {
                    if ((*_input)[b*vectorSize + i] > maximum)
                    {
                        maximum = (*_input)[b*vectorSize + i];
                    }
                }

                DataType expSum = 0;
                unsigned int label = (*_label)[b];

                for(unsigned int i=0;i<vectorSize;++i)
                {
                    DataType v = (*_input)[b*vectorSize + i] - maximum;

                    v = std::exp(v);

                    (*_output)[b*vectorSize + i] = v;

                    if (i == label)
                    {
                        (*_cost)[b] = v;
                    }

                    expSum += v;

                }

                for(unsigned int i=0;i<vectorSize;++i)
                {
                    (*_output)[b*vectorSize+i] = (*_output)[b*vectorSize+i] / expSum;
                }

                (*_cost)[b] /= expSum;

                (*_cost)[b] = -log((*_cost)[b]);
            
            }
        }
    };
}

#endif


