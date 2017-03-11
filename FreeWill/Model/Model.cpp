#include "Model.h"
#include <cmath>

FreeWill::Model* FreeWill::Model::create()
{
    FreeWill::Model *model = new FreeWill::Model();
    return model;
}

int FreeWill::Model::addTensor(const std::string &name, const Shape &shape, bool isBatchTensor)
{
    if (m_tensors.find(name) == m_tensors.end())
    {
        return -1;
    }
   
    m_tensors[name] = new FreeWill::Model::TensorDescriptor(name, shape, FLOAT, isBatchTensor);
    
   return 1;
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

FreeWill::Model::TensorDescriptor::TensorDescriptor(const std::__cxx11::string &name, const Shape &shape, DataType dataType, bool isBatchTensor)
    :m_name(name),
      m_shape(shape),
      m_isBatchTensor(isBatchTensor),
      m_dataType(dataType),
      m_tensors()
{

}

FreeWill::Model::TensorDescriptor::~TensorDescriptor()
{
    std::map<DeviceType, std::variant<TensorBase<GPU_CUDA>* >>::iterator iter = m_tensors.begin();
    for(;iter != m_tensors.end(); ++iter)
    {
        DeviceType deviceType = iter->first;

        switch(deviceType)
        {
        case CPU_NAIVE:
            break;
        case CPU_SIMD:
            break;
        case GPU_CUDA:
           
            break;
        }

    }
}

FreeWill::Model::Model(){}
