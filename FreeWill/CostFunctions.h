#ifndef COSTFUNCTIONS
#define COSTFUNCTIONS

#include <vector>
#include <cmath>
#include "ActivationFunctions.h"

template<class ScalarType>
void crossEntropySigmoidCPU(const ScalarType *outputs, unsigned int outputSize, const ScalarType *labels, ScalarType *cost, unsigned int batchSize, ScalarType *derivatives)
{
    for(int e = 0;e<batchSize;++e)
    {
        cost[e] = 0.0;
        for(size_t i = 0; i < outputSize; ++i)
        {
            ScalarType output = outputs[e * outputSize + i];
            ScalarType label = labels[e * outputSize + i];

            cost[e] += label*log(output) + (1.0 - label)*log(1.0 - output);
            derivatives[e*outputSize + i] = output - label;
        }
        cost[e] *= -1.0;
    }
}

template<class ScalarType>
void meanSquaredRectifierCPU(const ScalarType *outputs, unsigned int outputSize, const ScalarType *labels, ScalarType *cost, unsigned int batchSize, ScalarType *derivatives)
{
    ScalarType norm = 1.0 / outputSize;
    for(int e = 0;e<batchSize;++e)
    {
        cost[e] = 0.0;

        for(size_t i = 0; i<outputSize; ++i)
        {
            ScalarType output = outputs[e * outputSize + i];
            ScalarType label = labels[e * outputSize + i];

            cost[e] += (output - label) * (output - label);
            derivatives[e*outputSize + i] = output >=0 ? 2.0 * (output - label) * norm : 0.0;
        }
        cost[e] *= norm;
    }
}

//here outputs means last layer with activation but no cost function
/*template<class ScalarType>
void derivativeMeanSquaredTanh(const ScalarType *outputs, unsigned int outputSize, const ScalarType *labels, ScalarType *derivatives, unsigned int batchSize)
{

    for(int e = 0;e<batchSize;++e)
    {
        for(size_t i = 0; i < outputSize; ++i)
        {
            ScalarType tanhv = outputs[i * batchSize + e];
            derivatives[e*batchSize + i] = 2.0 * (tanhv - labels[i]) * (1.0 - tanhv * tanhv) / outputSize;
        }
    }
}

*/

/*
template<class ScalarType>
ScalarType derivativeMeanSquaredTanh2(ScalarType outputs, ScalarType labels)
{
    //derivatives.resize(outputs.size());

        ScalarType tanhv = tanhHelper(outputs);
        return 2.0 * (tanhv - labels) * (1.0 - tanhv * tanhv);
}*/




#endif // COSTFUNCTIONS

