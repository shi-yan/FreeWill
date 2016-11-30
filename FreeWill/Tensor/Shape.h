#ifndef SHAPE_H
#define SHAPE_H

#include <initializer_list>
#include <algorithm>

namespace FreeWill
{
    template<int Dimension>
    class Shape
    {
    private:
        unsigned int m_dim[Dimension] = {0};

    public:
        Shape()
        {
        }

        Shape(const Shape<Dimension> &shape)
        {
            *this = shape;
        }

        Shape(const unsigned int in[Dimension])
        {
            #pragma unroll
            for(int i = 0; i < Dimension; ++i)
            {
                m_dim[i] = in[i];
            }
        }

        Shape(const std::initializer_list<unsigned int> &li)
        {
            std::copy(li.begin(), li.begin() + std::min(li.size(), (unsigned long) Dimension), m_dim); 
        }

        unsigned int size() const 
        {
		    unsigned int size = 0;
            #pragma unroll
            for(int i = 0; i < Dimension; ++i)
            {
                size += m_dim[i];
            }
            return size;
        }

        void operator=(const Shape<Dimension> &shape)
        {
            #pragma unroll
            for(int i = 0; i < Dimension; ++i)
            {
                m_dim[i] = shape.m_dim[i];
            }
        }

        void operator=(const unsigned int in[Dimension])
        {
            #pragma unroll
            for(int i = 0; i < Dimension; ++i)
            {
                m_dim[i] = in[i];
            }
        }

        void operator=(const std::initializer_list<unsigned int> &li)
        {
            std::copy(li.begin(), li.begin() + std::min(li.size(), (unsigned long) Dimension), m_dim);
        }

        bool operator==(const Shape<Dimension> &shape) const 
        {
            #pragma unroll
            for(int i = 0; i < Dimension; ++i)
            {
                if (m_dim[i] != shape.m_dim[i])
                {
                    return false;
                }
            }
            return true;
        }

        bool operator!=(const Shape<Dimension> &shape) const
        {
            return !operator==(shape);
        }

        unsigned int &operator[](unsigned int i)
        {
            return m_dim[i];
        }
    };

    Shape<1> createShape(unsigned int size);
    Shape<2> createShape(unsigned int batchSize, unsigned int size);
    Shape<3> createShape(unsigned int batchSize, unsigned int height, unsigned int width);
}

#endif
