#ifndef OPERATOR_H
#define OPERATOR_H

#include <cstring>
#include "../Tensor/ReferenceCountedBlob.h"
#include "../DeviceSelection.h"
#include <map>
#include "../Tensor/Tensor.h"
#include <vector>

#include <cuda.h>
#include <cudnn.h>
#include <functional>
#include <variant>
#include <iostream>
#include <cxxabi.h>

namespace FreeWill
{
    template <DeviceType DeviceUsed>
    class OperatorFactory;

    typedef enum
    {
        ACTIVATION,
        ACTIVATION_DERIVATIVE,
        CONVOLUTION,
        CONVOLUTION_DERIVATIVE,
        CROSS_ENTROPY_LOSS,
        DOT_PRODUCT_WITH_BIAS,
        DOT_PRODUCT_WITH_BIAS_DERIVATIVE,
        ELEMENTWISE_ADD,
        MAX_POOLING,
        MAX_POOLING_DERIVATIVE,
        SIGMOID_CROSS_ENTROPY_LOSS_DERIVATIVE,
        SOFTMAX_LOG_LOSS,
        SOFTMAX_LOG_LOSS_DERIVATIVE
    } OperatorName;

    static std::map<std::string, OperatorName> operatorNameTable {{"Activation", ACTIVATION},
                {"ActivationDerivative", ACTIVATION_DERIVATIVE},
                {"Convolution", CONVOLUTION},
                {"ConvolutionDerivative", CONVOLUTION_DERIVATIVE},
                {"CrossEntropyLoss", CROSS_ENTROPY_LOSS},
                {"DotProductWithBias", DOT_PRODUCT_WITH_BIAS},
                {"ElementAdd", ELEMENTWISE_ADD},
                {"MaxPooling", MAX_POOLING},
                {"MaxPoolingDerivative", MAX_POOLING_DERIVATIVE},
                {"SigmoidCrossEntropyLossDerivative", SIGMOID_CROSS_ENTROPY_LOSS_DERIVATIVE},
                {"SoftmaxLogLoss", SOFTMAX_LOG_LOSS},
                {"SoftmaxLogLossDerivative", SOFTMAX_LOG_LOSS_DERIVATIVE}};

    template <DeviceType DeviceUsed = CPU>
    class Operator
    {
    protected:
        struct ParameterDescriptor
        {
            std::string m_name;
            std::vector<TensorBase<DeviceUsed>*> m_tensors;

            TensorBase<DeviceUsed> * operator[](unsigned int index)
            {
                return m_tensors[index];
            }
        };
        
        std::map<std::string, struct ParameterDescriptor > m_inputParameters;
        std::map<std::string, struct ParameterDescriptor > m_outputParameters;

    public:
        Operator() = delete;

        Operator(const std::initializer_list<std::string > &inputParameterList, 
                 const std::initializer_list<std::string > &outputParameterList)
        {
            typename std::initializer_list<std::string>::iterator iterInput = inputParameterList.begin();

            for(; iterInput != inputParameterList.end(); ++iterInput)
            {
                struct ParameterDescriptor d;
                d.m_name = (*iterInput);
                m_inputParameters[(*iterInput)] = d;
            }

           typename std::initializer_list<std::string>::iterator iterOutput = outputParameterList.begin(); 

           for (; iterOutput != outputParameterList.end(); ++iterOutput)
           {
                struct ParameterDescriptor d;
                d.m_name = (*iterOutput);
                m_outputParameters[(*iterOutput)] = d;
           }
        }
        
        virtual void evaluate() = 0;
        virtual bool init() = 0;
       
        virtual ~Operator(){};

        virtual int inputCount() 
        {
            typename std::map<std::string, struct ParameterDescriptor>::iterator iter = m_inputParameters.begin();

            unsigned int result = 0;
            for (; iter != m_inputParameters.end(); ++iter)
            {
                result += (*iter).second.m_tensors.size();
            }

            return result;
        }

        virtual int outputCount()
        {
            typename std::map<std::string, struct ParameterDescriptor>::iterator iter = m_outputParameters.begin();

            unsigned int result = 0;

            for (; iter != m_outputParameters.end(); ++iter)
            {
                result += (*iter).second.m_tensors.size();
            }

            return result;
        }

        virtual void setInputParameter(const std::string &name, TensorBase<DeviceUsed> *tensor)
        {
           if (m_inputParameters.find(name) != m_inputParameters.end())
           {
               m_inputParameters[name].m_tensors.push_back(tensor);
           }
           else 
           {
               printf("Warning: no input named %s\n", name.c_str());
           }
        }

        virtual void setOutputParameter(const std::string &name, TensorBase<DeviceUsed> *tensor)
        {
            if (m_outputParameters.find(name) != m_outputParameters.end())
            {
                m_outputParameters[name].m_tensors.push_back(tensor);
            }
            else
            {
                printf("Warning: no output named %s\n", name.c_str());
            }
        }

        virtual TensorBase<DeviceUsed> * input(const std::string &name, unsigned int index = 0)
        {
            if (m_inputParameters.find(name) != m_inputParameters.end())
            {
                if (m_inputParameters[name].m_tensors.size())
                {
                    return m_inputParameters[name].m_tensors[index];
                }
            }

            return 0;
        }

        virtual TensorBase<DeviceUsed> * output(const std::string &name, unsigned int index = 0)
        {
            if (m_outputParameters.find(name) != m_outputParameters.end())
            {
                if (m_outputParameters[name].m_tensors.size())
                {
                    return m_outputParameters[name].m_tensors[index];
                }
            }

            return 0;
        }

        virtual void clear()
        {
            typename std::map<std::string, struct ParameterDescriptor>::iterator iterInput = m_inputParameters.begin();

            for (; iterInput != m_inputParameters.end(); ++iterInput)
            {
                (*iterInput).second.m_tensors.clear();
            }

            typename std::map<std::string, struct ParameterDescriptor>::iterator iterOutput = m_outputParameters.begin();

            for (; iterOutput != m_outputParameters.end(); ++iterOutput)
            {
                (*iterOutput).second.m_tensors.clear();
            } 

        }

    };

    template <typename T, T /*unnamed*/>
    struct OperatorFactoryInitializer_ForceInit { };


    template <DeviceType DeviceUsed = CPU>
    class OperatorFactory
    {
        friend class Operator<DeviceUsed>;
    private:
        std::map<std::string, std::function<Operator<DeviceUsed>*(const std::map<std::string, std::variant<int, unsigned int, float, double>> &)>> m_creatFunctions;

        OperatorFactory()
            :m_creatFunctions(){};
        ~OperatorFactory(){};

    public:
        static OperatorFactory & getSingleton()
        {
            static OperatorFactory obj;
            return obj;
        }


    };

    template<typename OperatorType>
    class OperatorRegistry
    {
    public:
        class OperatorFactoryInitializer
        {
        public:
            int a;
            OperatorFactoryInitializer()
            {
                OperatorFactory<CPU>::getSingleton();
                char *realname;
                int status = 0;
                std::cout << "registered class:" << abi::__cxa_demangle(typeid(OperatorType).name(),0,0,&status) << std::endl;
                OperatorType::reg();
            }

            int getA()
            {
                return a;
            }

        };

        static OperatorFactoryInitializer m_operatorFactoryInitializer;
        typedef OperatorFactoryInitializer_ForceInit<OperatorFactoryInitializer&, m_operatorFactoryInitializer> __nnb_typedef_dummy__;
    };



    template<typename OperatorType> typename OperatorRegistry<OperatorType>::OperatorFactoryInitializer OperatorRegistry<OperatorType>::m_operatorFactoryInitializer;

   // template<>
   // Operator<CPU>::OperatorFactoryInitializer Operator<CPU>::m_operatorFactoryInitializer;



}
#endif
