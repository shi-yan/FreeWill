#ifndef WORD2VECCOSTFUNCTIONS_H
#define WORD2VECCOSTFUNCTIONS_H

#include <vector>
#include <cmath>

template<class ScalarType>
void softmaxCostAndGradient(const std::vector<ScalarType> &predicted, unsigned int target, const std::vector<std::vector<ScalarType>> &outputVectors, ScalarType &cost, std::vector<ScalarType> &gradPred, std::vector<std::vector<ScalarType>> &grad)
{

    std::vector<ScalarType> diff;
    std::vector<ScalarType> a;
    diff.resize(outputVectors.size(), 0.0);
    a.resize(outputVectors.size());
    ScalarType k = 0.0;

    for(int i = 0; i<outputVectors.size();++i)
    {

        for(int e = 0;e<predicted.size();++e)
        {
            diff[i]+= outputVectors[i][e] * predicted[e];
        }

        k+=(a[i] = std::exp(diff));
    }


    cost = std::log(k) - diff[target];

    std::vector<ScalarType> b;
    b.resize(a.size());
    for(int i = 0;i<b.size();++i)
    {
        b[i] = a[i]/k;
    }

    b[target] -= 1.0;

    gradPred.resize(predicted.size(), 0.0);
    for(int e = 0; e<gradPred.size();++e)
    {
        for(int i = 0; i<outputVectors.size();++i)
        {
            gradPred[e] += outputVectors[i][e] * b[i];
        }
    }

    grad.resize(outputVectors.size());

    for(int i = 0;i<grad.size();++i)
    {
        grad[i].resize(predicted.size());

        for(int e = 0;e<grad[i].size();++e)
        {
            grad[i][e] = b[i] * predicted[e];
        }
    }

}

#endif // WORD2VECCOSTFUNCTIONS_H
