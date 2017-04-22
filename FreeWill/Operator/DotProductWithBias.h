#ifndef DOTPRODUCTWITHBIAS_H
#define DOTPRODUCTWITHBIAS_H

#include "Operator.h"
//#include <QDebug>

#include <cublas_v2.h>
#include <type_traits>
#include "../Context/Context.h"


namespace FreeWill
{
    template<DeviceType DeviceUsed = DeviceType::CPU_NAIVE, typename DataType = float>
    class DotProductWithBias : public Operator<DeviceUsed>
    {
    protected:
        using Operator<DeviceUsed>::input;
        using Operator<DeviceUsed>::output;
        bool m_hasBias;
    public:
        DotProductWithBias(bool hasBias = true)
            :Operator<DeviceUsed>({"Input", "Weight","Bias"},{"Output"}),
            m_hasBias(hasBias)
        {
                
        }

        virtual bool init()
        {
            FAIL_IF(input("Input")==0 || input("Weight")==0 || output("Output") == 0);

            FAIL_IF ((input("Input")->shape().dimension() != 2) || 
                    (input("Weight")->shape().dimension() !=2) || 
                    (output("Output")->shape().dimension() != 2));

            unsigned int batchSize = input("Input")->shape()[1];
            unsigned int inputSize = input("Input")->shape()[0];
            unsigned int outputSize = output("Output")->shape()[0];

            FAIL_IF (batchSize != output("Output")->shape()[1] || batchSize == 0);
           
            //qDebug() << "weight" << input("Weight")->shape()[1];
            //qDebug() << "inputsize" << inputSize + 1;
            
            FAIL_IF(input("Weight")->shape()[1] != inputSize );
            

            FAIL_IF (input("Weight")->shape()[0]!= outputSize
                    || output("Output")->shape()[1] != batchSize);

            FAIL_IF (m_hasBias && input("Bias") == 0);

            if (m_hasBias)
            {
                FAIL_IF (input("Bias")->shape().dimension() != 1);
                              
                FAIL_IF (input("Bias")->shape()[0] != outputSize);
            }
            
            return true;
        }

        virtual void evaluate()
        {
            unsigned int batchSize = input("Input")->shape()[1];
            unsigned int inputSize = input("Input")->shape()[0];
            unsigned int outputSize = output("Output")->shape()[0];

            Tensor<DeviceUsed, DataType> *_input = input("Input")->template toType<DataType>();
            Tensor<DeviceUsed, DataType> *_weight = input("Weight")->template toType<DataType>();
            Tensor<DeviceUsed, DataType> *_output = output("Output")->template toType<DataType>();
            Tensor<DeviceUsed, DataType> *_bias = input("Bias")->template toType<DataType>();

            if constexpr (DeviceUsed == DeviceType::CPU_NAIVE)
            {
                for(unsigned int b = 0; b < batchSize; ++b)
                {
                    for(unsigned int o =0; o<outputSize;++o)
                    {
                        (*_output)[b * outputSize + o] = 0;
                        for(unsigned int i = 0; i< inputSize; ++i)
                        {
                            (*_output)[b * outputSize + o] += (*_weight)[i * outputSize + o] * (*_input)[b* inputSize + i];
                        }

                        if (m_hasBias)
                        {
                            (*_output)[b * outputSize + o] += (*_bias)[ o];
                        }
                    }
                }        
            }
            else if constexpr (DeviceUsed == DeviceType::GPU_CUDA)
            {
                DataType alpha = 1.0;
                DataType beta = 0.0;
                // input_t * weight_t = output_t
                // Trans(trans(input_t)*trans(weight_t)) = output_t
                // weight * input = output_t
                if constexpr (std::is_same<DataType, float>::value)
                {
                    
                    RUN_CUBLAS(cublasSgemm(Context<DeviceUsed>::getSingleton().cublasHandle(),CUBLAS_OP_N,CUBLAS_OP_N
                                ,outputSize, batchSize, inputSize, &alpha, _weight->gpuDataHandle(), outputSize, 
                                _input->gpuDataHandle(), inputSize, &beta, _output->gpuDataHandle(), outputSize));

                    if (m_hasBias)
                    {
                        beta = 1.0;
                        RUN_CUBLAS(cublasSgemm(Context<DeviceUsed>::getSingleton().cublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N,
                                    outputSize, batchSize, 1, &alpha, _bias->gpuDataHandle(), outputSize,
                                    Context<DeviceUsed>::getSingleton().template getSharedOneVector<DataType>(batchSize), 1, 
                                    &beta, _output->gpuDataHandle(), outputSize));
                    }

                }
                else if constexpr (std::is_same<DataType, double>::value)
                {
                    RUN_CUBLAS(cublasDgemm(Context<DeviceUsed>::getSingleton().cublasHandle(),CUBLAS_OP_N,CUBLAS_OP_N
                                ,outputSize, batchSize, inputSize, &alpha, _weight->gpuDataHandle(), outputSize,
                                _input->gpuDataHandle(), inputSize, &beta, _output->gpuDataHandle(), outputSize));

                    if (m_hasBias)
                    {
                        beta = 1.0;
                        RUN_CUBLAS(cublasDgemm(Context<DeviceUsed>::getSingleton().cublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N,
                                    outputSize, batchSize, 1, &alpha, _bias->gpuDataHandle(), outputSize,
                                    Context<DeviceUsed>::getSingleton().template getSharedOneVector<DataType>(batchSize), 1,
                                    &beta, _output->gpuDataHandle(), outputSize));
                    }

               }
            }
        }
    };
}


#endif
