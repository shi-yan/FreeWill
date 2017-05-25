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

    class Model;

    class TensorDescriptorHandle
    {
    private:
        Model *m_model;
        std::string m_name;
        Shape m_shape;
        bool m_isReshaped;

    public:
        TensorDescriptorHandle(Model *model = nullptr, const std::string &name = std::string(), const Shape &shape = Shape());
        TensorDescriptorHandle(const TensorDescriptorHandle &in)
            :m_model(in.m_model),
              m_name(in.m_name),
              m_shape(in.m_shape),
              m_isReshaped(in.m_isReshaped)
        {}

        void operator=(const TensorDescriptorHandle &in)
        {
            m_model = in.m_model;
            m_name = in.m_name;
            m_shape = in.m_shape;
            m_isReshaped = in.m_isReshaped;
        }
        TensorDescriptorHandle &enableBatch();
        TensorDescriptorHandle &randomize();

        const std::string &name() const
        {
            return m_name;
        }

        const Shape &shape() const
        {
            return m_shape;
        }

        bool isReshaped() const
        {
            return m_isReshaped;
        }

        TensorDescriptorHandle reshape(const Shape &newShape) const;
    };

    class TensorDescriptor
    {
    public:
        std::string m_name;
        Shape m_shape;
        bool m_isBatchTensor;
        int m_batchSize;
        bool m_isRandomlyInitialized;
        DataType m_dataType;

        std::map<DeviceType, std::vector<std::variant<TensorBase<DeviceType::GPU_CUDA>*, TensorBase<DeviceType::CPU_NAIVE>*>>> m_tensors;

        TensorDescriptor(const std::string &name, const Shape &shape, DataType dataType = DataType::FLOAT, bool isBatchTensor = false, bool isRandomlyInitialized = false);
        ~TensorDescriptor();

        void operator=(const TensorDescriptor &in);
        TensorDescriptor(const TensorDescriptor &in);

        bool isInitialized()
        {
            return !(m_tensors.size() == 0);
        }

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
        TensorBase<DeviceUsed> *getTensorForDevice(unsigned int deviceIndex)
        {
            return std::get<TensorBase<DeviceUsed>*>(m_tensors[DeviceUsed][deviceIndex]);
        }
    };
}

#endif
