#ifndef TENSOR_H
#define TENSOR_H

#include <type_traits>
#include <string>
#include <memory>
#include "DeviceSelection.h"
#include "Shape.h"

namespace FreeWill 
{
    template<int Dimension = 1, typename DataType = float>
    class Tensor_Common
    {
    protected:
	Shape<Dimension> m_shape;
        const std::string m_name;
        std::shared_ptr<DataType> m_data;

        Tensor_Common(const Shape<Dimension> &shape, const std::string &name)
            :m_shape(shape),
	        m_name(name)
        {}
    };


    template<int Dimension = 1, DeviceType DeviceUsed = CPU, typename DataType = float>
    class Tensor : public Tensor_Common<Dimension, DataType>
    {
    private:
        using Tensor_Common<Dimension, DataType>::m_shape;
        using Tensor_Common<Dimension, DataType>::m_name;
        using Tensor_Common<Dimension, DataType>::m_data;

    public:
        Tensor(const Shape<Dimension> &shape,
	       const std::string &name = "no_name")
            :Tensor_Common<Dimension, DataType>(shape, name)
	    {
	    }

        bool init()
	    {
	        if constexpr ((DeviceUsed & (CPU|CPU_SIMD)) != 0)
            {
            
            }
            else
            {
            
            }
	    }

        void randomiz()
        {
        
        }
    };
}

#endif
