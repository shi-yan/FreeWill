#ifndef SHAPE_H
#define SHAPE_H

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

        unsigned int getSize()
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

        bool operator==(const Shape<Dimension> &shape)
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

        unsigned int operator[](unsigned int i)
        {
            return m_dim[i];
        }
    };
}
#endif
