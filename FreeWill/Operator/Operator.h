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

#define FAIL_IF(EXP) \
    do { if (EXP) { \
             std::cerr << "Operator integrity check failed: " << #EXP << std::endl; \
             std::cerr << __FILE__ << ":"<< __LINE__ << std::endl; \
             Operator<DeviceUsed>::debugOutput(); \
             return false; \
    }} \
    while (0)

namespace FreeWill
{
    template <DeviceType DeviceUsed>
    class OperatorFactory;

    enum class OperatorName : uint32_t
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
        SOFTMAX_LOG_LOSS_DERIVATIVE,
        RESHAPE,
        DUPLICATE
    };

    static std::map<std::string, OperatorName> operatorNameTable {{"Activation", OperatorName::ACTIVATION},
                {"ActivationDerivative", OperatorName::ACTIVATION_DERIVATIVE},
                {"Convolution", OperatorName::CONVOLUTION},
                {"ConvolutionDerivative", OperatorName::CONVOLUTION_DERIVATIVE},
                {"CrossEntropyLoss", OperatorName::CROSS_ENTROPY_LOSS},
                {"DotProductWithBias", OperatorName::DOT_PRODUCT_WITH_BIAS},
                {"ElementAdd", OperatorName::ELEMENTWISE_ADD},
                {"MaxPooling", OperatorName::MAX_POOLING},
                {"MaxPoolingDerivative", OperatorName::MAX_POOLING_DERIVATIVE},
                {"SigmoidCrossEntropyLossDerivative", OperatorName::SIGMOID_CROSS_ENTROPY_LOSS_DERIVATIVE},
                {"SoftmaxLogLoss", OperatorName::SOFTMAX_LOG_LOSS},
                {"SoftmaxLogLossDerivative", OperatorName::SOFTMAX_LOG_LOSS_DERIVATIVE},
                {"Duplicate", OperatorName::DUPLICATE},
                {"Reshape", OperatorName::RESHAPE}};

    template <DeviceType DeviceUsed = DeviceType::CPU_NAIVE>
    class Operator
    {
    protected:
        struct ParameterDescriptor
        {
            std::string m_name;
            TensorBase<DeviceUsed>* m_tensor;
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
                d.m_tensor = nullptr;
                m_inputParameters[(*iterInput)] = d;
            }

           typename std::initializer_list<std::string>::iterator iterOutput = outputParameterList.begin(); 

           for (; iterOutput != outputParameterList.end(); ++iterOutput)
           {
                struct ParameterDescriptor d;
                d.m_name = (*iterOutput);
                d.m_tensor = nullptr;
                m_outputParameters[(*iterOutput)] = d;
           }
        }
        
        virtual void evaluate() = 0;
        virtual bool init() = 0;
       
        virtual ~Operator(){};

        void debugOutput()
        {
            std::cerr << "================= operator debug output =========================" << std::endl;
            typename std::map<std::string, struct ParameterDescriptor>::iterator iterInput = m_inputParameters.begin();

            for (; iterInput != m_inputParameters.end(); ++iterInput)
            {
                if (iterInput->second.m_tensor)
                {
                    std::cerr << "Input: " << iterInput->first;
                    std::cerr << " Tensor: " << iterInput->second.m_tensor->name();
                    std::cerr << " Shape: " << (iterInput->second.m_tensor->shape()) << std::endl;
                }
            }

            typename std::map<std::string, struct ParameterDescriptor>::iterator iterOutput = m_outputParameters.begin();

            for (; iterOutput != m_outputParameters.end(); ++iterOutput)
            {
                if (iterOutput->second.m_tensor)
                {
                    std::cerr << "Output: " << iterOutput->first;
                    std::cerr << " Tensor: " << iterOutput->second.m_tensor->name();
                    std::cerr << " Shape: " << (iterOutput->second.m_tensor->shape()) << std::endl;
                }
            }

            std::cerr << "================= ===================== =========================" << std::endl;

        }

        virtual int inputCount() 
        {
            return m_inputParameters.size();
        }

        virtual int outputCount()
        {
            return m_outputParameters.size();
        }

        virtual void setInputParameter(const std::string &name, TensorBase<DeviceUsed> *tensor)
        {
           if (m_inputParameters.find(name) != m_inputParameters.end())
           {
               m_inputParameters[name].m_tensor = tensor;
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
                m_outputParameters[name].m_tensor = tensor;
            }
            else
            {
                printf("Warning: no output named %s\n", name.c_str());
            }
        }

        virtual TensorBase<DeviceUsed> * input(const std::string &name)
        {
            if (m_inputParameters.find(name) != m_inputParameters.end())
            {
                return m_inputParameters[name].m_tensor;
            }

            return 0;
        }

        virtual TensorBase<DeviceUsed> * output(const std::string &name)
        {
            if (m_outputParameters.find(name) != m_outputParameters.end())
            {
                return m_outputParameters[name].m_tensor;
            }

            return 0;
        }

        virtual void clear()
        {
            typename std::map<std::string, struct ParameterDescriptor>::iterator iterInput = m_inputParameters.begin();

            for (; iterInput != m_inputParameters.end(); ++iterInput)
            {
                (*iterInput).second.m_tensor = nullptr;
            }

            typename std::map<std::string, struct ParameterDescriptor>::iterator iterOutput = m_outputParameters.begin();

            for (; iterOutput != m_outputParameters.end(); ++iterOutput)
            {
                (*iterOutput).second.m_tensor = nullptr;
            } 

        }

    };

    template <typename T, T /*unnamed*/>
    struct OperatorFactoryInitializer_ForceInit { };


    template <DeviceType DeviceUsed = DeviceType::CPU_NAIVE>
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
                OperatorFactory<DeviceType::CPU_NAIVE>::getSingleton();
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
