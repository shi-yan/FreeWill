#ifndef ACTIVATIONFUNCTIONS
#define ACTIVATIONFUNCTIONS

#include <cmath>

template<class ScalarType>
void sigmoid(const std::vector<ScalarType>& in, std::vector<ScalarType>& out)
{
    out.resize(in.size());
    for(size_t i = 0; i < in.size(); ++i)
    {
        out[i] = 1 / (1 + exp(-in[i]));
    }
}

template<class ScalarType>
ScalarType sigmoidDerivative(ScalarType in)
{
    return in * (1.0 - in);
}

template<class ScalarType>
void rectifier(const std::vector<ScalarType> &in, std::vector<ScalarType> &out)
{
    out.resize(in.size());
    for(size_t i = 0;i<in.size();++i)
    {
        out[i] = std::max((ScalarType)0.0f, (ScalarType)in[i]);
    }
}

template<class ScalarType>
ScalarType rectifierDerivative(ScalarType in)
{
    if (in > 0)
    {
        return 1;
    }
    else return 0;
}


template<class ScalarType>
void softmax(const std::vector<ScalarType>& in, std::vector<ScalarType>& out)
{
    ScalarType average = 0.0;
    foreach(ScalarType zElement, in)
    {
        average += zElement;
    }

    average /= in.size();

    out.resize(in.size());

    ScalarType b = 0.0;
    for(int i = 0; i < in.size(); ++i)
    {
        out[i] = exp(in[i] - average);
        b += out[i];
    }

    b = 1.0 / b;

    for(int i = 0; i < out.size(); ++i)
    {
        out[i] *= b;
    }
}

template<class ScalarType>
void noActivation(const std::vector<ScalarType>& in, std::vector<ScalarType>& out)
{
    out = in;
}

#endif // ACTIVATIONFUNCTIONS

