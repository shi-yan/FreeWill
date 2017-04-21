#include "Model.h"
#include <cmath>
#include "../Operator/Operator.h"

FreeWill::Model* FreeWill::Model::create()
{
    FreeWill::Model *model = new FreeWill::Model();
    return model;
}

FreeWill::TensorDescriptorHandle FreeWill::Model::addTensor(const std::string &name, const Shape &shape, bool isBatchTensor, DataType dataType)
{
    if (m_tensors.find(name) != m_tensors.end())
    {
        return {std::string(), Shape()};
    }
   
    m_tensors[name] = new FreeWill::TensorDescriptor(name, shape, isBatchTensor, dataType);
    
   return {name, Shape()};
}



FreeWill::Model::Model()
    :m_tensors(),
      m_operators()
{
}

int FreeWill::Model::addOperator(const std::string &name,
                                 const std::string &operatorNameString,
                                 const std::map<std::string, FreeWill::TensorDescriptorHandle> &inputs,
                                 const std::map<std::string, FreeWill::TensorDescriptorHandle> &outputs,
                                 const std::map<std::string, std::any> &properties, DataType dataType)
{
    if (FreeWill::operatorNameTable.find(operatorNameString) != FreeWill::operatorNameTable.end())
    {
        FreeWill::OperatorName operatorName = operatorNameTable[operatorNameString];

        addOperator(name, operatorName, inputs, outputs, properties, dataType);
    }

    return -1;
}

int FreeWill::Model::addOperator(const std::string &name,
                                 FreeWill::OperatorName operatorName,
                                 const std::map<std::string, FreeWill::TensorDescriptorHandle> &inputs,
                                 const std::map<std::string, FreeWill::TensorDescriptorHandle> &outputs,
                                 const std::map<std::string, std::any> &properties, DataType dataType)
{
    if (m_operators.find(name) != m_operators.end())
    {
        return -1;
    }

    OperatorDescriptor *opDescriptor = new OperatorDescriptor(name, operatorName, inputs, outputs, properties, dataType);

    m_operators[name] = opDescriptor;

    return m_operators.size() - 1;
}



bool FreeWill::Model::init(const Solver &solver)
{
    //allocating tensors
    std::map<std::string, TensorDescriptor*>::iterator iterTensor = m_tensors.begin();

    //creating operators

    std::map<std::string, OperatorDescriptor*>::iterator iterOperator = m_operators.begin();

    switch (solver.m_deviceUsed)
    {
    case CPU_NAIVE:
        for(;iterTensor != m_tensors.end(); ++iterTensor)
        {
            std::cout << iterTensor->first << std::endl;
            TensorDescriptor *descriptor = iterTensor->second;
            descriptor->allocateTensor<FreeWill::CPU_NAIVE>(solver);
        }

        for(;iterOperator != m_operators.end(); ++iterOperator)
        {
            std::cout << iterOperator->first << std::endl;
            OperatorDescriptor *descriptor = iterOperator->second;

            if (!descriptor->init<FreeWill::CPU_NAIVE>(m_tensors))
            {
                return false;
            }
        }

        break;
    case GPU_CUDA:

        for(;iterTensor != m_tensors.end(); ++iterTensor)
        {
            std::cout << iterTensor->first << std::endl;
            TensorDescriptor *descriptor = iterTensor->second;
            descriptor->allocateTensor<FreeWill::GPU_CUDA>(solver);
        }

        for(;iterOperator != m_operators.end(); ++iterOperator)
        {
            std::cout << iterOperator->first << std::endl;
            OperatorDescriptor *descriptor = iterOperator->second;

            if (!descriptor->init<FreeWill::GPU_CUDA>(m_tensors))
            {
                std::cout << iterOperator->first << "sanity check failed!" << std::endl;
                return false;
            }
        }

        break;
    default:
        return false;
    }

    return true;
}

