#include "TensorDescriptor.h"

FreeWill::TensorDescriptor::TensorDescriptor(const TensorDescriptor &in)
    :m_name(in.m_name),
      m_shape(in.m_shape),
      m_isBatchTensor(in.m_isBatchTensor),
      m_batchSize(in.m_batchSize),
      m_dataType(in.m_dataType),
      m_tensors(in.m_tensors)
{
}

void FreeWill::TensorDescriptor::operator =(const TensorDescriptor &in)
{
    m_name = in.m_name;
    m_shape = in.m_shape;
    m_isBatchTensor = in.m_isBatchTensor;
    m_batchSize = in.m_batchSize;
    m_dataType = in.m_dataType;
    m_tensors = in.m_tensors;
}

FreeWill::TensorDescriptor::TensorDescriptor(const std::__cxx11::string &name, const Shape &shape, bool isBatchTensor, DataType dataType)
    :m_name(name),
      m_shape(shape),
      m_isBatchTensor(isBatchTensor),
      m_batchSize(0),
      m_dataType(dataType),
      m_tensors()
{

}

FreeWill::TensorDescriptor::~TensorDescriptor()
{
    std::map<DeviceType, std::vector<std::variant<TensorBase<GPU_CUDA>* ,TensorBase<CPU_NAIVE>* >>>::iterator iter = m_tensors.begin();
    for(;iter != m_tensors.end(); ++iter)
    {
        DeviceType deviceType = iter->first;

        std::vector<std::variant<TensorBase<GPU_CUDA>* ,TensorBase<CPU_NAIVE>* >> &tensorList = iter->second;

        switch(deviceType)
        {
        case CPU_NAIVE:
        {
            for(auto iter = tensorList.begin(); iter != tensorList.end(); ++iter)
            {
                TensorBase<CPU_NAIVE> *tensor = std::get<TensorBase<CPU_NAIVE>*>(*iter);
                delete tensor;
            }
        }
            break;
        case GPU_CUDA:
        {
            for(auto iter = tensorList.begin(); iter != tensorList.end(); ++iter)
            {
                TensorBase<GPU_CUDA> *tensor = std::get<TensorBase<GPU_CUDA>*>(*iter);
                delete tensor;
            }
        }
            break;
        }

        tensorList.clear();

    }
}


FreeWill::TensorDescriptorHandle operator^(FreeWill::TensorDescriptorHandle &handle, const FreeWill::Shape &newShape)
{
    FreeWill::TensorDescriptorHandle newHandle = handle;
    newHandle.second = newShape;
    return newHandle;
}
