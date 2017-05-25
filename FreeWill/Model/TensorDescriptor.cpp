#include "TensorDescriptor.h"
#include "Model.h"

FreeWill::TensorDescriptor::TensorDescriptor(const TensorDescriptor &in)
    :m_name(in.m_name),
      m_shape(in.m_shape),
      m_isBatchTensor(in.m_isBatchTensor),
      m_batchSize(in.m_batchSize),
      m_isRandomlyInitialized(in.m_isRandomlyInitialized),
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
    m_isRandomlyInitialized = in.m_isRandomlyInitialized;
    m_dataType = in.m_dataType;
    m_tensors = in.m_tensors;
}

FreeWill::TensorDescriptor::TensorDescriptor(const std::string &name, const Shape &shape, DataType dataType, bool isBatchTensor, bool isRandomlyInitialized)
    :m_name(name),
      m_shape(shape),
      m_isBatchTensor(isBatchTensor),
      m_batchSize(0),
      m_isRandomlyInitialized(isRandomlyInitialized),
      m_dataType(dataType),
      m_tensors()
{

}

FreeWill::TensorDescriptor::~TensorDescriptor()
{
    std::map<DeviceType, std::vector<std::variant<TensorBase<DeviceType::GPU_CUDA>* ,TensorBase<DeviceType::CPU_NAIVE>* >>>::iterator iter = m_tensors.begin();
    for(;iter != m_tensors.end(); ++iter)
    {
        DeviceType deviceType = iter->first;

        std::vector<std::variant<TensorBase<DeviceType::GPU_CUDA>* ,TensorBase<DeviceType::CPU_NAIVE>* >> &tensorList = iter->second;

        switch(deviceType)
        {
        case DeviceType::CPU_NAIVE:
        {
            for(auto iter = tensorList.begin(); iter != tensorList.end(); ++iter)
            {
                TensorBase<DeviceType::CPU_NAIVE> *tensor = std::get<TensorBase<DeviceType::CPU_NAIVE>*>(*iter);
                delete tensor;
            }
        }
            break;
        case DeviceType::GPU_CUDA:
        {
            for(auto iter = tensorList.begin(); iter != tensorList.end(); ++iter)
            {
                TensorBase<DeviceType::GPU_CUDA> *tensor = std::get<TensorBase<DeviceType::GPU_CUDA>*>(*iter);
                delete tensor;
            }
        }
            break;
        }

        tensorList.clear();

    }
}


/*FreeWill::TensorDescriptorHandle operator^(FreeWill::TensorDescriptorHandle &handle, const FreeWill::Shape &newShape)
{
    FreeWill::TensorDescriptorHandle newHandle = handle;
    newHandle.m_shape = newShape;
    return newHandle;
}*/

FreeWill::TensorDescriptorHandle::TensorDescriptorHandle(Model *model, const std::string &name, const Shape &shape)
    :m_model(model),
      m_name(name),
      m_shape(shape),
      m_isReshaped(false)
{}

FreeWill::TensorDescriptorHandle &FreeWill::TensorDescriptorHandle::enableBatch()
{
    TensorDescriptor* tensorDescriptor = m_model->m_tensors[m_name];

    if (!tensorDescriptor->isInitialized())
    {
        tensorDescriptor->m_isBatchTensor = true;
    }
    else
    {
        std::cerr << "Can't enable batch after tensor is initialized!"<<std::endl;
    }

    return *this;
}

FreeWill::TensorDescriptorHandle &FreeWill::TensorDescriptorHandle::randomize()
{
    TensorDescriptor* tensorDescriptor = m_model->m_tensors[m_name];

    if (!tensorDescriptor->isInitialized())
    {
        tensorDescriptor->m_isRandomlyInitialized = true;
    }
    else
    {
        std::cerr << "Can't set randomization after tensor is initialized!"<<std::endl;
    }
    return *this;
}

FreeWill::TensorDescriptorHandle FreeWill::TensorDescriptorHandle::reshape(const FreeWill::Shape &newShape) const
{
    FreeWill::TensorDescriptorHandle newHandle = *this;
    newHandle.m_shape = newShape;
    newHandle.m_isReshaped = true;
    return newHandle;
}
