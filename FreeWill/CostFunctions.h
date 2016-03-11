#ifndef COSTFUNCTIONS
#define COSTFUNCTIONS

#include <vector>
#include <cmath>

template<class ScalarType>
void crossEntropy(const std::vector<ScalarType> &outputs, const std::vector<ScalarType> &labels, ScalarType &cost)
{
    cost = 0.0;
    for(size_t i = 0; i < outputs.size(); ++i)
    {
        cost += labels[i]*log(outputs[i]) + (1 - labels[i])*log(1 - outputs[i]);
    }
    cost *= -1.0;
}

template<class ScalarType>
void derivativeCrossEntropySigmoid(const std::vector<ScalarType> &outputs, const std::vector<ScalarType> &labels, std::vector<ScalarType> &derivatives)
{
    derivatives.resize(outputs.size());

    for(size_t i = 0; i < outputs.size(); ++i)
    {
        derivatives[i] = outputs[i] - labels[i];
    }
}

template<class ScalarType>
void meanSquared(const std::vector<ScalarType> &outputs, const std::vector<ScalarType> &labels, ScalarType &cost)
{
    cost = 0.0;
    for(size_t i = 0;i<outputs.size();++i)
    {
        cost += (outputs[i] - labels[i])*(outputs[i] - labels[i]);
    }
    cost /= (float) outputs.size();
}

template<class ScalarType>
void derivativeMeanSquaredRectifier(const std::vector<ScalarType> &outputs, const std::vector<ScalarType> &labels, std::vector<ScalarType> &derivatives)
{
    derivatives.resize(outputs.size());

    for(size_t i = 0; i < outputs.size(); ++i)
    {
        derivatives[i] = 2.0 * (outputs[i] - labels[i]) / outputs.size();
    }
}


#endif // COSTFUNCTIONS

