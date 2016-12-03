#ifndef SHAPE_H
#define SHAPE_H

#include <initializer_list>
#include <algorithm>

namespace FreeWill
{
    class Shape
    {
    private:
        unsigned int *m_dim;
        unsigned int m_size;

    public:
        Shape(unsigned int size)
            :m_dim(nullptr),
            m_size(0)
        {
            m_size = size;
            m_dim = new unsigned int[m_size];
        }

        Shape()
            :m_dim(nullptr),
             m_size(0)
        {
        }

        Shape(const Shape &shape)
            :m_dim(nullptr),
            m_size(0)
        {
            *this = shape;
        }

        Shape(const unsigned int *in, unsigned int size)
            :m_dim(nullptr),
            m_size(0)
        {
            delete [] m_dim;
            m_dim = new unsigned int [size];
            m_size = size;
            #pragma unroll
            for(unsigned int i = 0; i < size; ++i)
            {
                m_dim[i] = in[i];
            }
        }

        Shape(const std::initializer_list<unsigned int> &li)
            :m_dim(nullptr),
             m_size(0)
        {
            *this = li;
        }

        unsigned int size() const 
        {
		    unsigned int size = 1;
            #pragma unroll
            for(unsigned int i = 0; i < m_size; ++i)
            {
                size *= m_dim[i];
            }
            return size;
        }

        unsigned int dimension() const
        {
            return m_size;
        }

        void operator=(const Shape &shape)
        {
            delete [] m_dim;
            m_dim = new unsigned int[shape.m_size];
            m_size = shape.m_size;
            #pragma unroll
            for(unsigned int i = 0; i < m_size; ++i)
            {
                m_dim[i] = shape.m_dim[i];
            }
        }

        void operator=(const std::initializer_list<unsigned int> &li)
        {
            delete [] m_dim;
            m_dim = new unsigned int[li.size()];
            m_size = li.size();
            std::copy(li.begin(), li.begin() + m_size, m_dim); 
        }

        bool operator==(const Shape &shape) const 
        {
            if (m_size != shape.m_size)
            {
                return false;
            }
            #pragma unroll
            for(unsigned int i = 0; i < m_size; ++i)
            {
                if (m_dim[i] != shape.m_dim[i])
                {
                    return false;
                }
            }
            return true;
        }

        bool operator!=(const Shape &shape) const
        {
            return !operator==(shape);
        }

        unsigned int &operator[](unsigned int i)
        {
            return m_dim[i];
        }

        unsigned int operator[](unsigned int i) const
        {
            return m_dim[i];
        }

        ~Shape()
        {
            delete [] m_dim;
            m_size = 0;
        }
    };
}

#endif
