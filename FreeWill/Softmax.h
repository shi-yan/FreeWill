#ifndef SOFTMAX
#define SOFTMAX

#include <cmath>

template<class ScalarType, class VectorType>
VectorType softmax(VectorType z)
{
    ScalarType average = 0.0;
    foreach(ScalarType zElement, z)
    {
        average += zElement;
    }

    average /= z.size();

    VectorType result;

    ScalarType b = 0.0;
    for(int i = 0; i < z.size(); ++i)
    {
        ScalarType c = exp(z[i] - average);
        result.push_back(c);
        b += c;
    }

    for(int i = 0; i < z.size(); ++i)
    {
        result[i] /= b;
    }

    return result;
}

#endif // SOFTMAX

