#ifndef TENSOR_H
#define TENSOR_H

#include <type_traits>
#include <string>
#include <memory>
#include "DeviceSelection.h"
#include "Shape.h"
#include "ReferenceCountedBlob.h"


namespace FreeWill 
{
    template<DeviceType DeviceUsed>
    class TensorBase
    {
    protected:
       ReferenceCountedBlob<DeviceUsed> m_data;
       TensorBase() 
       {}

       TensorBase(const ReferenceCountedBlob<DeviceUsed> &data)
           :m_data(data)
       {}

    public:
       virtual ~TensorBase() {}
    };
    
    template<DeviceType DeviceUsed = CPU, typename DataType = float>
    class Tensor : public TensorBase<DeviceUsed>
    {
    private:
        Shape m_shape;
        std::string m_name;
        using TensorBase<DeviceUsed>::m_data;
        
    public:
        explicit Tensor(const Shape &shape = Shape(),
	       const std::string &name = "no_name")
            :TensorBase<DeviceUsed>(),
            m_shape(shape),
            m_name(name)
	    {
	    }

        explicit Tensor(const Tensor &in)
            :TensorBase<DeviceUsed>(in.m_data),
            m_shape(in.m_shape),
            m_name(in.m_name)
        {
        }

        bool init()
	    {
            unsigned int size = m_shape.size();
            if (size) 
            {
                return m_data.alloc(size * sizeof(DataType));
            }
	    }

        void randomize()
        {
            if constexpr ((DeviceUsed & (CPU_SIMD | CPU_NAIVE)) != 0)
            {
                 std::random_device rd;
                 std::mt19937 gen(rd());
                 std::uniform_real_distribution<DataType> dis(0, 1);
                 DataType *bits = (DataType *) m_data.dataHandle();
                 unsigned int size = m_shape.size();
                 for (unsigned int n = 0; n < size; ++n) 
                 {
                     bits[n] = dis(gen);
                 } 
            }
            else
            {
            
            }
        }

        void operator=(const Tensor<DeviceUsed, DataType> &in)
        {
            m_shape = in.m_shape;
            m_name = in.m_name;
            m_data = in.m_data;
        }

        const Shape & shape() const
        {
            return m_shape;
        }

        DataType &operator[](unsigned int i)
        {
            DataType *bits = (DataType *) m_data.dataHandle();
            return *(bits + i);
        }
    };
}

#endif
