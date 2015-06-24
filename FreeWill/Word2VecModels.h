#ifndef WORD2VECMODELS
#define WORD2VECMODELS

#include <string>
#include <functional>
#include <map>
#include <vector>
#include "Word2VecCostFunctions.h"

template <class ScalarType>
void skipgram(const std::string &currentWord,
              unsigned int C,
              const std::vector<std::string> &contextWords,
              const std::map<std::string, unsigned int> &tokens,
              const std::vector<std::vector<ScalarType>> &inputVectors,
              const std::vector<std::vector<ScalarType>> &outputVectors,
              std::function<void(const std::vector<ScalarType> &, unsigned int, const std::vector<std::vector<ScalarType>> &, ScalarType &, std::vector<ScalarType> &, std::vector<std::vector<ScalarType>> &)> word2vecCostAndGradient,
              ScalarType cost, std::vector<std::vector<ScalarType>> &inputGrad, std::vector<std::vector<ScalarType>> &outputGrad)
{

    std::vector<ScalarType> r = inputVectors[tokens.at(currentWord)];

    ScalarType overallCost = 0.0;
    std::vector<ScalarType> overallGradPred;
    overallGradPred.resize(r.size, 0.0);

    outputGrad.resize(outputVectors.size());
    for(int i = 0; i< outputGrad.size(); ++i)
    {
        outputGrad[i].resize(r.size(), 0.0);
    }

    ScalarType eachCost = 0.0;
    cost = 0.0;
    std::vector<ScalarType> gradPred;
    std::vector<std::vector<ScalarType>> grad;

    for (int h = 0;h<contextWords.size();++h)
    {
        word2vecCostAndGradient(r, tokens.at(contextWords[h]), outputVectors, eachCost, gradPred, grad);
        cost += eachCost;

        for(int i = 0;i<gradPred.size();++i)
        {
            overallGradPred[i] += gradPred[i];
        }

        for(int i = 0;i<outputGrad.size();++i)
        {
            for(int e = 0;e<outputGrad.size();++e)
            {
                outputGrad[i][e] += grad[i][e];
            }
        }
    }

    inputGrad.resize(inputVectors.size());
    for(int i = 0; i< inputGrad.size(); ++i)
    {
        inputGrad[i].resize(r.size(), 0.0);
    }

    inputGrad[tokens.at(currentWord)] =  overallGradPred;

}

template<class ScalarType>
void CBOW(const std::string &currentWord,
          unsigned int C,
          const std::vector<std::string> &contextWords,
          const std::map<std::string, unsigned int> &tokens,
          const std::vector<std::vector<ScalarType>> &inputVectors,
          const std::vector<std::vector<ScalarType>> &outputVectors,
          std::function<void(const std::vector<ScalarType> &, unsigned int, const std::vector<std::vector<ScalarType>> &, ScalarType &, std::vector<ScalarType> &, std::vector<std::vector<ScalarType>> &)> word2vecCostAndGradient,
          ScalarType cost, std::vector<std::vector<ScalarType>> &inputGrad, std::vector<std::vector<ScalarType>> &outputGrad)
{
    std::vector<ScalarType> r;
    r.resize(inputVectors[0].size(), 0);

    for(int i =0;i<contextWords.size();++i)
    {
        for(int e =0;e<r.size();++e)
        {
            r[e] += inputVectors[tokens.at(contextWords[i])][e];
        }
    }

    for(int e =0;e<r.size();++e)
    {
        r[e] /= C;
    }

    std::vector<ScalarType> gradPred;

    word2vecCostAndGradient(r, tokens.at(currentWord), outputVectors, cost, gradPred, outputGrad );

    inputGrad.resize(inputVectors.size());
    for(int i = 0; i< inputGrad.size(); ++i)
    {
        inputGrad[i].resize(r.size(), 0.0);
    }



    for(int i = 0;i<contextWords.size();++i)
    {
        for(int e = 0;e<gradPred.size();++e)
        {
            inputGrad[tokens.at(contextWords[i])][e] += gradPred[e] / C;
        }
    }
}

#endif // WORD2VECMODELS

