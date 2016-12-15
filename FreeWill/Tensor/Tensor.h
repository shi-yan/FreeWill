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

       Shape m_shape;

       ReferenceCountedBlob<DeviceUsed> m_data;
       TensorBase(const Shape &shape = Shape()) 
           :m_shape(shape),
            m_data()
       {}

       TensorBase(const ReferenceCountedBlob<DeviceUsed> &data, const Shape &shape = Shape())
           :m_shape(shape),
               m_data(data)
       {}
    public:
       virtual const Shape &shape() const
       {
            return m_shape;
       }

       void clear()
       {
        m_data.clear();
       }

       virtual ~TensorBase() {}
    };
    
    template<DeviceType DeviceUsed = CPU, typename DataType = float>
    class Tensor : public TensorBase<DeviceUsed>
    {
    private:
        using TensorBase<DeviceUsed>::m_shape;
        std::string m_name;
        using TensorBase<DeviceUsed>::m_data;
        
    public:
        using TensorBase<DeviceUsed>::shape;

        
        explicit Tensor(const Shape &shape = Shape(),
	       const std::string &name = "no_name")
            :TensorBase<DeviceUsed>(shape),
            m_name(name)
	    {
	    }

        explicit Tensor(const Tensor &in)
            :TensorBase<DeviceUsed>(in.m_data, in.shape()),
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

            return false;
	    }

        bool init(const std::initializer_list<DataType> &initList)
        {
            if (!init())
            {
                return false;
            }

            unsigned int size = m_shape.size();
            std::copy(initList.begin(), initList.begin() + (initList.size()>size?size:initList.size()), (DataType*) m_data.dataHandle());
            return true;
        }

        void randomize()
        {
            if constexpr ((DeviceUsed & (CPU_SIMD | CPU_NAIVE)) != 0)
            {
                 //std::random_device rd;
                 //std::mt19937 gen(rd());
                 //std::uniform_real_distribution<DataType> dis(0, 1);
                 DataType *bits = (DataType *) m_data.dataHandle();
                 unsigned int size = m_shape.size();
                 for (unsigned int n = 0; n < size; ++n) 
                 {
                     //bits[n] = dis(gen);
                     bits[n] = (double) rand() / (double) RAND_MAX;
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

        DataType &operator[](unsigned int i)
        {
            DataType *bits = (DataType *) m_data.dataHandle();
            return *(bits + i);
        }

        void reshape(const Shape &newShape)
        {
            if (newShape.size() == m_shape.size())
            {
                m_shape = newShape;
            }
        }
    };
}

#endif
