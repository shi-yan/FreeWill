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
    template<int Dimension = 1, DeviceType DeviceUsed = CPU, typename DataType = float>
    class Tensor
    {
    private:
        Shape<Dimension> m_shape;
        std::string m_name;
        ReferenceCountedBlob<DeviceUsed, DataType> m_data;

    public:
        Tensor(const Shape<Dimension> &shape,
	       const std::string &name = "no_name")
            :m_shape(shape),
            m_name(name),
            m_data()
	    {
	    }

        bool init()
	    {
            unsigned int size = m_shape.size();
            return m_data.alloc(size);
	    }

        void randomize()
        {
            m_data.randomize();
        }
    };
}

#endif
