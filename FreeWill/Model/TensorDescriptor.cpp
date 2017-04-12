#include "TensorDescriptor.h"

FreeWill::TensorDescriptor::TensorDescriptor(const TensorDescriptor &in)
    :m_name(in.m_name),
      m_shape(in.m_shape),
      m_batchSize(in.m_batchSize),
      m_dataType(in.m_dataType),
      m_tensors(in.m_tensors)
{
}

void FreeWill::TensorDescriptor::operator =(const TensorDescriptor &in)
{
    m_name = in.m_name;
    m_shape = in.m_shape;
    m_batchSize = in.m_batchSize;
    m_dataType = in.m_dataType;
    m_tensors = in.m_tensors;
}

FreeWill::TensorDescriptor::TensorDescriptor(const std::__cxx11::string &name, const Shape &shape, bool isBatchTensor, DataType dataType)
    :m_name(name),
      m_shape(shape),
      m_isBatchTensor(isBatchTensor),
      m_dataType(dataType),
      m_tensors()
{

}

FreeWill::TensorDescriptor::~TensorDescriptor()
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
