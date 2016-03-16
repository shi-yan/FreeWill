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
    if (in >= 0)
    {
        return 1;
    }
    else return 0;
}

template<class ScalarType>
ScalarType tanhHelper(ScalarType x)
{
  ScalarType y = exp(2.0 * x);
  return (y - 1) / (y + 1);
}

template<class ScalarType>
void tanh(const std::vector<ScalarType> &in, std::vector<ScalarType> &out)
{
    out.resize(in.size());
    for(size_t i = 0;i<in.size();++i)
    {
        out[i] = tanhHelper<ScalarType>(in[i]);
    }
}

//this in is after activation tanh
template<class ScalarType>
ScalarType tanhDerivative(ScalarType in)
{
    ScalarType tanhv = in;
    return (1.0 - tanhv * tanhv);
    //return (1.0 - in * in);
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

template<class ScalarType>
ScalarType noActivationDerivative(ScalarType in)
{
    return 1;
}

#endif // ACTIVATIONFUNCTIONS

