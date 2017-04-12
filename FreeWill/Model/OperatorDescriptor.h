#ifndef OPERATORDESCRIPTOR_H
#define OPERATORDESCRIPTOR_H

#include "../Operator/Operator.h"
#include "../Operator/Activation.h"
#include "../Operator/ActivationDerivative.h"
#include "../Operator/Convolution.h"
#include "../Operator/ConvolutionDerivative.h"
#include "../Operator/CrossEntropyLoss.h"
#include "../Operator/DotProductWithBias.h"
#include "../Operator/DotProductWithBiasDerivative.h"
#include "../Operator/ElementwiseAdd.h"
#include "../Operator/MaxPooling.h"
#include "../Operator/MaxPoolingDerivative.h"
#include "../Operator/SigmoidCrossEntropyLossDerivative.h"
#include "../Operator/SoftmaxLogLoss.h"
#include "../Operator/SoftmaxLogLossDerivative.h"
#include "TensorDescriptor.h"
#include <any>

namespace FreeWill
{

    class Model;
    class OperatorDescriptor
    {
        friend class Model;
    private:
        std::string m_name;
        DataType m_dataType;
        OperatorName m_operatorName;
        std::map<std::string, FreeWill::TensorDescriptorHandle> m_inputs;
        std::map<std::string, FreeWill::TensorDescriptorHandle> m_outputs;
        std::map<std::string, std::any> m_parameters;

        OperatorDescriptor(const std::string &name, OperatorName operatorName,
                           const std::map<std::string, FreeWill::TensorDescriptorHandle> &inputs,
                           const std::map<std::string, FreeWill::TensorDescriptorHandle> &outputs,
                           const std::map<std::string, std::any> &parameters, DataType dataType = FLOAT);
        ~OperatorDescriptor();

        std::map<DeviceType, std::variant<Operator<GPU_CUDA>*, Operator<CPU_NAIVE>*>> m_operators;


        template<DeviceType DeviceUsed>
        Operator<DeviceUsed> *initActivation()
        {
            Operator<DeviceUsed> *operatorBase = nullptr;
            if (m_parameters.find("Mode") == m_parameters.end())
            {
                return nullptr;
            }

            FreeWill::ActivationMode mode = m_parameters["Mode"];
            switch(mode)
            {
            case SIGMOID:
                switch(m_dataType)
                {
                case FLOAT:
                    operatorBase = new Activation<SIGMOID, DeviceUsed, float>();
                    break;
                case DOUBLE:
                    operatorBase = new Activation<SIGMOID, DeviceUsed, double>();
                    break;
                case UNSIGNED_INT:
                    operatorBase = new Activation<SIGMOID, DeviceUsed, unsigned int>();
                    break;
                }
                break;
            case RELU:
                switch(m_dataType)
                {
                case FLOAT:
                    operatorBase = new Activation<RELU, DeviceUsed, float>();
                    break;
                case DOUBLE:
                    operatorBase = new Activation<RELU, DeviceUsed, double>();
                    break;
                case UNSIGNED_INT:
                    operatorBase = new Activation<RELU, DeviceUsed, unsigned int>();
                    break;
                }
                break;
            case TANH:
                switch(m_dataType)
                {
                case FLOAT:
                    operatorBase = new Activation<TANH, DeviceUsed, float>();
                    break;
                case DOUBLE:
                    operatorBase = new Activation<TANH, DeviceUsed, double>();
                    break;
                case UNSIGNED_INT:
                    operatorBase = new Activation<TANH, DeviceUsed, unsigned int>();
                    break;
                }
                break;
            case CLIPPED_RELU:
                switch(m_dataType)
                {
                case FLOAT:
                    operatorBase = new Activation<CLIPPED_RELU, DeviceUsed, float>();
                    break;
                case DOUBLE:
                    operatorBase = new Activation<CLIPPED_RELU, DeviceUsed, double>();
                    break;
                case UNSIGNED_INT:
                    operatorBase = new Activation<CLIPPED_RELU, DeviceUsed, unsigned int>();
                    break;
                }
                break;
            }

            return operatorBase;
        }

        template<DeviceType DeviceUsed>
        Operator<DeviceUsed> *initActivationDerivative()
        {
            Operator<DeviceUsed> *operatorBase = nullptr;
            if (m_parameters.find("Mode") == m_parameters.end())
            {
                return nullptr;
            }

            FreeWill::ActivationMode mode = m_parameters["Mode"];
            switch(mode)
            {
            case SIGMOID:
                switch(m_dataType)
                {
                case FLOAT:
                    operatorBase = new ActivationDerivative<SIGMOID, DeviceUsed, float>();
                    break;
                case DOUBLE:
                    operatorBase = new ActivationDerivative<SIGMOID, DeviceUsed, double>();
                    break;
                case UNSIGNED_INT:
                    operatorBase = new ActivationDerivative<SIGMOID, DeviceUsed, unsigned int>();
                    break;
                }
                break;
            case RELU:
                switch(m_dataType)
                {
                case FLOAT:
                    operatorBase = new ActivationDerivative<RELU, DeviceUsed, float>();
                    break;
                case DOUBLE:
                    operatorBase = new ActivationDerivative<RELU, DeviceUsed, double>();
                    break;
                case UNSIGNED_INT:
                    operatorBase = new ActivationDerivative<RELU, DeviceUsed, unsigned int>();
                    break;
                }
                break;
            case TANH:
                switch(m_dataType)
                {
                case FLOAT:
                    operatorBase = new ActivationDerivative<TANH, DeviceUsed, float>();
                    break;
                case DOUBLE:
                    operatorBase = new ActivationDerivative<TANH, DeviceUsed, double>();
                    break;
                case UNSIGNED_INT:
                    operatorBase = new ActivationDerivative<TANH, DeviceUsed, unsigned int>();
                    break;
                }
                break;
            case CLIPPED_RELU:
                switch(m_dataType)
                {
                case FLOAT:
                    operatorBase = new ActivationDerivative<CLIPPED_RELU, DeviceUsed, float>();
                    break;
                case DOUBLE:
                    operatorBase = new ActivationDerivative<CLIPPED_RELU, DeviceUsed, double>();
                    break;
                case UNSIGNED_INT:
                    operatorBase = new ActivationDerivative<CLIPPED_RELU, DeviceUsed, unsigned int>();
                    break;
                }
                break;
            }

            return operatorBase;
        }

        template<DeviceType DeviceUsed>
        Operator<DeviceUsed> *initConvolution()
        {
            Operator<DeviceUsed> *operatorBase = nullptr;

            switch(m_dataType)
            {
            case FLOAT:
                operatorBase = new Convolution<DeviceUsed, float>();

                break;
            case DOUBLE:
                operatorBase = new Convolution<DeviceUsed, double>();

                break;
            case UNSIGNED_INT:
                operatorBase = new Convolution<DeviceUsed, unsigned int>();
                break;

            }

            return operatorBase;
        }

        template<DeviceType DeviceUsed>
        Operator<DeviceUsed> *initConvolutionDerivative()
        {
            Operator<DeviceUsed> *operatorBase = nullptr;

            switch(m_dataType)
            {
            case FLOAT:
                operatorBase = new ConvolutionDerivative<DeviceUsed, float>();
                break;
            case DOUBLE:
                operatorBase = new ConvolutionDerivative<DeviceUsed, double>();
                break;
            case UNSIGNED_INT:
                operatorBase = new ConvolutionDerivative<DeviceUsed, unsigned int>();
                break;

            }

            return operatorBase;

        }

        template<DeviceType DeviceUsed>
        Operator<DeviceUsed> *initCrossEntropyLoss()
        {
            Operator<DeviceUsed> *operatorBase = nullptr;

            switch(m_dataType)
            {
            case FLOAT:
                operatorBase = new CrossEntropyLoss<DeviceUsed, float>();
                break;
            case DOUBLE:
                operatorBase = new CrossEntropyLoss<DeviceUsed, double>();
                break;
            case UNSIGNED_INT:
                operatorBase = new CrossEntropyLoss<DeviceUsed, unsigned int>();
                break;
            }

            return operatorBase;
        }


        template<DeviceType DeviceUsed>
        Operator<DeviceUsed> *initDotProductWithBias()
        {
            Operator<DeviceUsed> *operatorBase = nullptr;

            switch(m_dataType)
            {
            case FLOAT:
                operatorBase = new DotProductWithBias<DeviceUsed, float>();

                break;
            case DOUBLE:
                operatorBase = new DotProductWithBias<DeviceUsed, double>();
                break;
            case UNSIGNED_INT:
                operatorBase = new DotProductWithBias<DeviceUsed, unsigned int>();
                break;
            }

            return operatorBase;

        }

        template<DeviceType DeviceUsed>
        Operator<DeviceUsed> *initDotProductWithBiasDerivative()
        {
            Operator<DeviceUsed> *operatorBase = nullptr;

            switch(m_dataType)
            {
            case FLOAT:
                operatorBase = new DotProductWithBiasDerivative<DeviceUsed, float>();
                break;
            case DOUBLE:
                operatorBase = new DotProductWithBiasDerivative<DeviceUsed, double>();
                break;
            case UNSIGNED_INT:
                operatorBase = new DotProductWithBiasDerivative<DeviceUsed, unsigned int>();

                break;
            }

            return operatorBase;
        }

        template<DeviceType DeviceUsed>
        Operator<DeviceUsed> *initElementwiseAdd()
        {
            Operator<DeviceUsed> *operatorBase = nullptr;

            switch(m_dataType)
            {
            case FLOAT:
                operatorBase = new ElementwiseAdd<DeviceUsed, float>();
                break;
            case DOUBLE:
                operatorBase = new ElementwiseAdd<DeviceUsed, double>();
                break;
            case UNSIGNED_INT:
                operatorBase = new ElementwiseAdd<DeviceUsed, unsigned int>();
                break;
            }

            return operatorBase;
        }

        template<DeviceType DeviceUsed>
        Operator<DeviceUsed> *initMaxPooling()
        {
            Operator<DeviceUsed> *operatorBase = nullptr;

            switch(m_dataType)
            {
            case FLOAT:
                operatorBase = new MaxPooling<DeviceUsed, float>();
                break;
            case DOUBLE:
                operatorBase = new MaxPooling<DeviceUsed, double>();
                break;
            case UNSIGNED_INT:
                operatorBase = new MaxPooling<DeviceUsed, unsigned int>();
                break;


            }

            return operatorBase;

        }

        template<DeviceType DeviceUsed>
        Operator<DeviceUsed> *initMaxPoolingDerivative()
        {
            Operator<DeviceUsed> *operatorBase = nullptr;

            switch(m_dataType)
            {
            case FLOAT:
                operatorBase = new MaxPoolingDerivative<DeviceUsed, float>();
                break;
            case DOUBLE:
                operatorBase = new MaxPoolingDerivative<DeviceUsed, double>();
                break;
            case UNSIGNED_INT:
                operatorBase = new MaxPoolingDerivative<DeviceUsed, unsigned int>();
                break;
            }

            return operatorBase;
        }

        template<DeviceType DeviceUsed>
        Operator<DeviceUsed> *initSigmoidCrossEntropyLossDerivative()
        {
            Operator<DeviceUsed> *operatorBase = nullptr;

            switch(m_dataType)
            {
            case FLOAT:
                operatorBase = new SigmoidCrossEntropyLossDerivative<DeviceUsed, float>();
                break;
            case DOUBLE:
                operatorBase = new SigmoidCrossEntropyLossDerivative<DeviceUsed, double>();
                break;
            case UNSIGNED_INT:
                operatorBase = new SigmoidCrossEntropyLossDerivative<DeviceUsed, unsigned int>();

                break;
            }

            return operatorBase;
        }

        template<DeviceType DeviceUsed>
        Operator<DeviceUsed> *initSoftmaxLogLoss()
        {
            Operator<DeviceUsed> *operatorBase = nullptr;

            switch (m_dataType)
            {
            case FLOAT:
                operatorBase = new SoftmaxLogLoss<DeviceUsed, float>();
                break;
            case DOUBLE:
                operatorBase = new SoftmaxLogLoss<DeviceUsed, double>();
                break;
            case UNSIGNED_INT:
                operatorBase = new SoftmaxLogLoss<DeviceUsed, unsigned int>();
                break;
            }

            return operatorBase;
        }

        template<DeviceType DeviceUsed>
        Operator<DeviceUsed> *initSoftmaxLogLossDerivative()
        {
            Operator<DeviceUsed> *operatorBase = nullptr;

            switch(m_dataType)
            {
            case FLOAT:
                operatorBase = new SoftmaxLogLossDerivative<DeviceUsed, float>();
                break;
            case DOUBLE:
                operatorBase = new SoftmaxLogLossDerivative<DeviceUsed, double>();
                break;
            case UNSIGNED_INT:
                operatorBase = new SoftmaxLogLossDerivative<DeviceUsed, unsigned int>();
                break;
            }

            return operatorBase;
        }

        template<DeviceType DeviceUsed = CPU_NAIVE>
        bool init()
        {
            Operator<DeviceUsed> *operatorBase = nullptr;

            switch(m_operatorName)
            {
            case ACTIVATION:
                operatorBase = initActivation<DeviceUsed>();
            break;
            case ACTIVATION_DERIVATIVE:
                operatorBase = initActivationDerivative<DeviceUsed>();
            break;
            case CONVOLUTION:
                operatorBase = initConvolution<DeviceUsed>();
            break;
            case CONVOLUTION_DERIVATIVE:
                operatorBase = initConvolutionDerivative<DeviceUsed>();
            break;
            case CROSS_ENTROPY_LOSS:
                operatorBase = initCrossEntropyLoss<DeviceUsed>();
            break;
            case DOT_PRODUCT_WITH_BIAS:
                operatorBase = initDotProductWithBias<DeviceUsed>();
            break;
            case DOT_PRODUCT_WITH_BIAS_DERIVATIVE:
                operatorBase = initDotProductWithBiasDerivative<DeviceUsed>();
            break;
            case ELEMENTWISE_ADD:
                operatorBase = initElementwiseAdd<DeviceUsed>();
            break;
            case MAX_POOLING:
                operatorBase = initMaxPooling<DeviceUsed>();
            break;
            case MAX_POOLING_DERIVATIVE:
                operatorBase = initMaxPoolingDerivative<DeviceUsed>();
            break;
            case SIGMOID_CROSS_ENTROPY_LOSS_DERIVATIVE:
                operatorBase = initSigmoidCrossEntropyLossDerivative<DeviceUsed>();
            break;
            case SOFTMAX_LOG_LOSS:
                operatorBase = initSoftmaxLogLoss<DeviceUsed>();
            break;
            case SOFTMAX_LOG_LOSS_DERIVATIVE:
                operatorBase = initSoftmaxLogLossDerivative<DeviceUsed>();
            break;
            }

            if (!operatorBase)
            {
                return false;
            }

            return true;
        }
    };
}

#endif
