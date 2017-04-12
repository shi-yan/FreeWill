#ifndef OPERATORDESCRIPTOR_H
#define OPERATORDESCRIPTOR_H

#include "../Operator/Operator.h"
#include "../Operator/Activation.h"
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

        template<DeviceType DeviceUsed = CPU_NAIVE>
        bool init()
        {
            Operator<DeviceUsed> *operatorBase = nullptr;

            switch(m_operatorName)
            {
            case ACTIVATION:
                if (operatorBase = initActivation<DeviceUsed>())
                {
                    return true;
                }
            break;
        case ACTIVATION_DERIVATIVE:
            break;
        case CONVOLUTION:
            break;
        case CONVOLUTION_DERIVATIVE:
            break;
        case CROSS_ENTROPY_LOSS:
            break;
        case DOT_PRODUCT_WITH_BIAS:
            break;
        case DOT_PRODUCT_WITH_BIAS_DERIVATIVE:
            break;
        case ELEMENTWISE_ADD:
            break;
        case MAX_POOLING:
            break;
        case MAX_POOLING_DERIVATIVE:
            break;
        case SIGMOID_CROSS_ENTROPY_LOSS_DERIVATIVE:
            break;
        case SOFTMAX_LOG_LOSS:
            break;
        case SOFTMAX_LOG_LOSS_DERIVATIVE:
            break;
        }
    }
};
}

#endif
