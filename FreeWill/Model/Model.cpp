#include "Model.h"
#include <cmath>
#include "../Operator/Operator.h"

FreeWill::Model* FreeWill::Model::create()
{
    FreeWill::Model *model = new FreeWill::Model();
    return model;
}

FreeWill::Model::TensorDescriptorHandle FreeWill::Model::addTensor(const std::string &name, const Shape &shape, bool isBatchTensor, DataType dataType)
{
    if (m_tensors.find(name) == m_tensors.end())
    {
        return std::string();
    }
   
    m_tensors[name] = new FreeWill::Model::TensorDescriptor(name, shape, isBatchTensor, dataType);
    
   return name;
}

FreeWill::Model::TensorDescriptor::TensorDescriptor(const TensorDescriptor &in)
    :m_name(in.m_name),
      m_shape(in.m_shape),
      m_batchSize(in.m_batchSize),
      m_dataType(in.m_dataType),
      m_tensors(in.m_tensors)
{
}

void FreeWill::Model::TensorDescriptor::operator =(const TensorDescriptor &in)
{
    m_name = in.m_name;
    m_shape = in.m_shape;
    m_batchSize = in.m_batchSize;
    m_dataType = in.m_dataType;
    m_tensors = in.m_tensors;
}

FreeWill::Model::TensorDescriptor::TensorDescriptor(const std::__cxx11::string &name, const Shape &shape, bool isBatchTensor, DataType dataType)
    :m_name(name),
      m_shape(shape),
      m_isBatchTensor(isBatchTensor),
      m_dataType(dataType),
      m_tensors()
{

}

FreeWill::Model::TensorDescriptor::~TensorDescriptor()
{
    std::map<DeviceType, std::variant<TensorBase<GPU_CUDA>* ,TensorBase<CPU_NAIVE>* >>::iterator iter = m_tensors.begin();
    for(;iter != m_tensors.end(); ++iter)
    {
        DeviceType deviceType = iter->first;

        switch(deviceType)
        {
        case CPU_NAIVE:
        {
            TensorBase<CPU_NAIVE> *tensor = std::get<TensorBase<CPU_NAIVE>*>(iter->second);
            delete tensor;
        }
            break;
        case GPU_CUDA:
        {
            TensorBase<GPU_CUDA> *tensor = std::get<TensorBase<GPU_CUDA>*>(iter->second);
            delete tensor;
        }
            break;
        }

    }
}

FreeWill::Model::Model()
    :m_tensors(),
      m_operators()
{
}

int FreeWill::Model::addOperator(const std::string &name,
                                 const std::string &operatorNameString,
                                 const std::map<std::string, FreeWill::Model::TensorDescriptorHandle> &inputs,
                                 const std::map<std::string, FreeWill::Model::TensorDescriptorHandle> &outputs,
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
                                 const std::map<std::string, FreeWill::Model::TensorDescriptorHandle> &inputs,
                                 const std::map<std::string, FreeWill::Model::TensorDescriptorHandle> &outputs,
                                 const std::map<std::string, std::any> &properties, DataType dataType)
{
    if (m_operators.find(name) != m_operators.end())
    {
        return -1;
    }

    OperatorDescriptor *opDescriptor = new OperatorDescriptor(name, operatorName, properties, dataType);

    m_operators[name] = opDescriptor;

    return m_operators.size() - 1;
}

FreeWill::Model::OperatorDescriptor::OperatorDescriptor(const std::string &name,
        FreeWill::OperatorName operatorName,
        const std::map<std::string, std::any> &parameters, 
        DataType dataType)
    :m_name(name),
      m_operatorName(operatorName),
      m_parameters(parameters),
      m_dataType(dataType)
{

}

FreeWill::Model::OperatorDescriptor::~OperatorDescriptor()
{

}

