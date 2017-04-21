#include "Shape.h"
#include <cstdio>

namespace FreeWill
{
	Shape operator+(const Shape &in, unsigned int batchSize)
    {
        if (batchSize > 0)
        {
            Shape shape;

            shape.m_dim = new unsigned int[in.dimension() + 1];
            std::copy(in.m_dim, in.m_dim + in.dimension(), shape.m_dim);
            shape.m_dim[in.dimension()] = batchSize;
            shape.m_count = in.dimension() + 1;

            return shape;
        }

        return in;
    }

    std::ostream& operator<< (std::ostream& stream, Shape const &shape)
    {
        stream << shape.dimension();
        stream << " {";

        for (unsigned int i =0;i<shape.m_count;++i)
        {
            stream << shape.m_dim[i] << ((i==(shape.m_count-1))?"}":", ");
        }

        return stream;
    }
}
