#ifndef TENSORDESCRIPTOR_H
#define TENSORDESCRIPTOR_H

#include "../Tensor/Tensor.h"
#include "../Tensor/Shape.h"
#include <map>
#include <variant>

namespace FreeWill
{

    typedef enum
    {
        FLOAT,
        DOUBLE,
        UNSIGNED_INT
    } DataType;

    typedef std::string TensorDescriptorHandle;

    class Model;

    class TensorDescriptor
    {
        friend class Model;
    public:
        std::string m_name;
        Shape m_shape;
        bool m_isBatchTensor;
        int m_batchSize;
        DataType m_dataType;
        //change this to variant or any
        std::map<DeviceType, std::variant<TensorBase<GPU_CUDA>*, TensorBase<CPU_NAIVE>*>> m_tensors;

        TensorDescriptor(const std::string &name, const Shape &shape, bool isBatchTensor = false, DataType dataType = FLOAT);
        ~TensorDescriptor();

        void operator=(const TensorDescriptor &in);
        TensorDescriptor(const TensorDescriptor &in);

        template<DeviceType DeviceUsed = CPU_NAIVE>
        void allocateTensor()
        {
            FreeWill::TensorBase<DeviceUsed> *tensor = nullptr;
            switch (m_dataType)
            {
            case FLOAT:
                tensor = new FreeWill::Tensor<DeviceUsed, float>(m_shape, m_name);
                tensor->template toType<float>()->init();
                break;
            case DOUBLE:
                tensor = new FreeWill::Tensor<DeviceUsed, double>(m_shape, m_name);
                tensor->template toType<double>()->init();
                break;
            case UNSIGNED_INT:
                tensor = new FreeWill::Tensor<DeviceUsed, unsigned int>(m_shape, m_name);
                tensor->template toType<unsigned int>()->init();
                break;
            default:
                break;
            }


            m_tensors[DeviceUsed] = tensor;
        }
    };
}
#endif
