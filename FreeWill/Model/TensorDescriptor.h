#ifndef TENSORDESCRIPTOR_H
#define TENSORDESCRIPTOR_H

#include "../Tensor/Tensor.h"
#include "../Tensor/Shape.h"
#include <map>
#include <variant>
#include "Solver.h"

namespace FreeWill
{

    typedef enum
    {
        FLOAT,
        DOUBLE,
        UNSIGNED_INT
    } DataType;

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
        DataType m_dataType;
        //change this to variant or any
        std::map<DeviceType, std::vector<std::variant<TensorBase<GPU_CUDA>*, TensorBase<CPU_NAIVE>*>>> m_tensors;

        TensorDescriptor(const std::string &name, const Shape &shape, bool isBatchTensor = false, DataType dataType = FLOAT);
        ~TensorDescriptor();

        void operator=(const TensorDescriptor &in);
        TensorDescriptor(const TensorDescriptor &in);

        template<DeviceType DeviceUsed = CPU_NAIVE>
        void allocateTensor(const Solver &solver)
        {
            FreeWill::TensorBase<DeviceUsed> *tensor = nullptr;
            switch (m_dataType)
            {
            case FLOAT:
                tensor = new FreeWill::Tensor<DeviceUsed, float>(m_isBatchTensor?(m_shape + (m_batchSize = solver.m_batchSize)):m_shape, m_name);
                tensor->template toType<float>()->init();
                break;
            case DOUBLE:
                tensor = new FreeWill::Tensor<DeviceUsed, double>(m_isBatchTensor?(m_shape + (m_batchSize = solver.m_batchSize)):m_shape, m_name);
                tensor->template toType<double>()->init();
                break;
            case UNSIGNED_INT:
                tensor = new FreeWill::Tensor<DeviceUsed, unsigned int>(m_isBatchTensor?(m_shape + (m_batchSize = solver.m_batchSize)):m_shape, m_name);
                tensor->template toType<unsigned int>()->init();
                break;
            default:
                break;
            }


            m_tensors[DeviceUsed].push_back(tensor);
        }

        template<DeviceType DeviceUsed = CPU_NAIVE>
        TensorBase<DeviceUsed> *getTensorForDevice(unsigned int deviceIndex = 0)
        {
            return std::get<TensorBase<DeviceUsed>*>(m_tensors[DeviceUsed][deviceIndex]);
        }
    };
}

FreeWill::TensorDescriptorHandle operator^(FreeWill::TensorDescriptorHandle &handle, const FreeWill::Shape &newShape);

#endif
