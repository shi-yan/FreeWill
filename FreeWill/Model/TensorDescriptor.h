#ifndef TENSORDESCRIPTOR_H
#define TENSORDESCRIPTOR_H

#include "../Tensor/Tensor.h"
#include "../Tensor/Shape.h"
#include <map>
#include <variant>
#include <cstdint>
#include "../Context/Context.h"

namespace FreeWill
{

    enum class DataType : uint32_t
    {
        FLOAT,
        DOUBLE,
        UNSIGNED_INT
    };

    typedef std::pair<std::string, Shape> TensorDescriptorHandle;

    class Model;

    class TensorDescriptor
    {
        friend class Model;
    public:
        std::string m_name;
        Shape m_shape;
        bool m_isBatchTensor;
        int m_batchSize;
        bool m_isRandomlyInitialized;
        DataType m_dataType;

        std::map<DeviceType, std::vector<std::variant<TensorBase<DeviceType::GPU_CUDA>*, TensorBase<DeviceType::CPU_NAIVE>*>>> m_tensors;

        TensorDescriptor(const std::string &name, const Shape &shape, bool isBatchTensor = false, bool isRandomlyInitialized = true, DataType dataType = DataType::FLOAT);
        ~TensorDescriptor();

        void operator=(const TensorDescriptor &in);
        TensorDescriptor(const TensorDescriptor &in);

        template<DeviceType DeviceUsed = DeviceType::CPU_NAIVE>
        void allocateTensor(unsigned int batchSize)
        {
            int deviceCount = Context<DeviceUsed>::getSingleton().deviceCount();

            for (int i =0;i<deviceCount;++i)
            {
                FreeWill::TensorBase<DeviceUsed> *tensor = nullptr;
                switch (m_dataType)
                {
                case DataType::FLOAT:
                    tensor = new FreeWill::Tensor<DeviceUsed, float>(m_isBatchTensor?(m_shape + (m_batchSize = batchSize)):m_shape, m_name);
                    tensor->template toType<float>()->init();
                    if (m_isRandomlyInitialized)
                    {
                        if (i == 0)
                        {
                            tensor->template toType<float>()->randomize();
                        }
                        else
                        {
                            TensorBase<DeviceUsed> *firstTensor = std::get<TensorBase<DeviceUsed>*>(m_tensors[DeviceUsed][0]);
                            std::copy((unsigned char*)firstTensor->cpuDataHandle(),
                                      ((unsigned char*)firstTensor->cpuDataHandle())+firstTensor->sizeInByte(), (unsigned char*) tensor->cpuDataHandle());
                        }
                    }
                    break;
                case DataType::DOUBLE:
                    tensor = new FreeWill::Tensor<DeviceUsed, double>(m_isBatchTensor?(m_shape + (m_batchSize = batchSize)):m_shape, m_name);
                    tensor->template toType<double>()->init();
                    if (m_isRandomlyInitialized)
                    {
                        if (i == 0)
                        {
                            tensor->template toType<double>()->randomize();
                        }
                        else
                        {
                            TensorBase<DeviceUsed> *firstTensor = std::get<TensorBase<DeviceUsed>*>(m_tensors[DeviceUsed][0]);
                            std::copy((unsigned char*)firstTensor->cpuDataHandle(),
                                      ((unsigned char*)firstTensor->cpuDataHandle())+firstTensor->sizeInByte(), (unsigned char*) tensor->cpuDataHandle());
                        }
                    }
                    break;
                case DataType::UNSIGNED_INT:
                    tensor = new FreeWill::Tensor<DeviceUsed, unsigned int>(m_isBatchTensor?(m_shape + (m_batchSize = batchSize)):m_shape, m_name);
                    tensor->template toType<unsigned int>()->init();
                    if (m_isRandomlyInitialized)
                    {
                        //tensor->template toType<unsigned int>()->randomize();
                    }
                    break;
                default:
                    break;
                }

                m_tensors[DeviceUsed].push_back(tensor);
            }
        }

        template<DeviceType DeviceUsed = DeviceType::CPU_NAIVE>
        TensorBase<DeviceUsed> *getTensorForDevice(unsigned int deviceIndex = 0)
        {
            return std::get<TensorBase<DeviceUsed>*>(m_tensors[DeviceUsed][deviceIndex]);
        }
    };
}

FreeWill::TensorDescriptorHandle operator^(FreeWill::TensorDescriptorHandle &handle, const FreeWill::Shape &newShape);

#endif
