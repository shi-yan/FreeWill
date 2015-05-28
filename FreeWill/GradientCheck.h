#ifndef GRADIENTCHECK
#define GRADIENTCHECK

#include <functional>

template<class InputType, class OutputType>
OutputType gradientCheck(std::function<OutputType(InputType)> function, InputType x, InputType epsilon)
{
    OutputType gradient = (function(x + epsilon) - function(x - epsilon)) / (2.0 * epsilon);
    return gradient;
}

#endif // GRADIENTCHECK

