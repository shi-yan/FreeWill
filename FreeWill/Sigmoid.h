#ifndef SIGMOID
#define SIGMOID

#include <cmath>

template<class ValueType>
ValueType sigmoid(ValueType x)
{
    ValueType sigmoidValue = 1 / (1 + exp(-x));
    return sigmoidValue;
}

#endif // SIGMOID

