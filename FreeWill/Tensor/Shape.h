#ifndef SHAPE_H
#define SHAPE_H

#include <initializer_list>
#include <algorithm>
#include <string>
#include <sstream>

namespace FreeWill
{
    class Shape
    {
    private:
        unsigned int *m_dim;
        unsigned int m_count;

    public:
        Shape(unsigned int dimension = 0)
            :m_dim(nullptr),
            m_count(dimension)
        {
            if (m_count)
            {
                m_dim = new unsigned int[m_count];
            }
        }

        Shape(const Shape &shape)
            :m_dim(nullptr),
            m_count(0)
        {
            *this = shape;
        }

        Shape(const unsigned int *in, unsigned int count)
            :m_dim(nullptr),
            m_count(0)
        {
            delete [] m_dim;
            m_dim = new unsigned int [count];
            m_count = count;

            for(unsigned int i = 0; i < count; ++i)
            {
                m_dim[i] = in[i];
            }
        }

        Shape(const std::initializer_list<unsigned int> &li)
            :m_dim(nullptr),
             m_count(0)
        {
            *this = li;
        }

        unsigned int size() const 
        {
            unsigned int size = 1;

            for(unsigned int i = 0; i < m_count; ++i)
            {
                size *= m_dim[i];
            }
            return size;
        }

        unsigned int dimension() const
        {
            return m_count;
        }

        void operator=(const Shape &shape)
        {
            delete [] m_dim;
            m_dim = new unsigned int[shape.m_count];
            m_count = shape.m_count;

            for(unsigned int i = 0; i < m_count; ++i)
            {
                m_dim[i] = shape.m_dim[i];
            }
        }

        void operator=(const std::initializer_list<unsigned int> &li)
        {
            delete [] m_dim;
            m_dim = new unsigned int[li.size()];
            m_count = li.size();
            std::copy(li.begin(), li.begin() + m_count, m_dim);
        }

        bool operator==(const Shape &shape) const 
        {
            if (m_count != shape.m_count)
            {
                return false;
            }
            for(unsigned int i = 0; i < m_count; ++i)
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
            m_count = 0;
        }

        friend
        Shape operator+(const Shape &in, unsigned int batchSize);

        std::string toString()
        {
            std::stringstream stream;
            stream << dimension();
            stream << " {";

            for (unsigned int i =0;i<m_count;++i)
            {
                stream << m_dim[i] << ((i==m_count-1)?"}":", ");
            }

            return stream.str();
        }

        friend std::ostream& operator<< (std::ostream& stream, Shape const &shape);
    };

    Shape operator+(const Shape &in, unsigned int batchSize);
    std::ostream& operator<< (std::ostream& stream, Shape const &shape);
}

#endif
