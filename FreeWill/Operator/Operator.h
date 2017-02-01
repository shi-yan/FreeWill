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


namespace FreeWill
{
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
}
#endif
