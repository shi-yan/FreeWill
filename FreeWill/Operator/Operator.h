#ifndef OPERATOR_H
#define OPERATOR_H

#include <cstring>
#include "../Tensor/ReferenceCountedBlob.h"
#include "../DeviceSelection.h"
#include <map>
#include "../Tensor/Tensor.h"
#include <vector>


namespace FreeWill
{
    template <DeviceType DeviceUsed = CPU>
    class Operator
    {
    protected:
        enum Direction
        {
            Input,
            Output
        };

        struct ParameterDescriptor
        {
            std::string m_name;
            enum Direction m_direction;
            std::vector<TensorBase<DeviceUsed> *> m_tensorList;
            unsigned int m_min;
            unsigned int m_max;
        };
        
        struct ParameterInitDescriptor
        {
            std::string m_name;
            enum Direction m_direction;
            unsigned int m_min;
            unsigned int m_max;
        };

        std::map<std::string, struct ParameterDescriptor > parameters;

    public:
        Operator(std::initializer_list<struct ParameterInitDescriptor > &parameterList)
        {
/*            std::initializer_list<struct ParameterInitDescriptor>::iter iter = parameterList.begin();

            for(; iter != parameterList.end(); ++iter)
            {
                struct ParameterDescriptor d;
                d.m_name = (*iter).m_name;
                d.m_direction = (*iter).m_direction;
                d.m_min = (*iter).m_min;
                d.m_max = (*iter).m_max;
                parameters[(*iter).m_name] = d;
            }            
  */      }
        Operator(){}
        virtual void evaluate() = 0;
        virtual bool init() = 0;
       
      
        virtual ~Operator(){};

        virtual int inputCount() {return 0;};
        virtual int outputCount() {return 0;};

        virtual void setParameter(const std::string &name, TensorBase<DeviceUsed> *tensor)
        {
           if (parameters.find(name) != parameters.end())
           {
               parameters[name].m_tensorList.push_back(tensor);
           } 
        }
    };
}
#endif
