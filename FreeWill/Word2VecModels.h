#ifndef WORD2VECMODELS
#define WORD2VECMODELS

#include <string>
#include <functional>
#include <map>
#include <vector>
#include "Word2VecCostFunctions.h"
#include "Word2VecDataset.h"
#include "StanfordSentimentDataset.h"

template <class ScalarType>
void skipgram(const std::string &currentWord,
              unsigned int C,
              const std::vector<std::string> &contextWords,
              const std::map<std::string, unsigned int> &tokens,
              const std::vector<std::vector<ScalarType>> &inputVectors,
              const std::vector<std::vector<ScalarType>> &outputVectors,
              std::function<void(const std::vector<ScalarType> &, unsigned int, const std::vector<std::vector<ScalarType>> &, ScalarType &, std::vector<ScalarType> &, std::vector<std::vector<ScalarType>> &)> word2vecCostAndGradient,
              ScalarType &cost, std::vector<std::vector<ScalarType>> &inputGrad, std::vector<std::vector<ScalarType>> &outputGrad)
{

    std::vector<ScalarType> r = inputVectors[tokens.at(currentWord)];

    ScalarType overallCost = 0.0;
    std::vector<ScalarType> overallGradPred;
    overallGradPred.resize(r.size(), 0.0);

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
            for(int e = 0;e<outputGrad[0].size();++e)
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
          ScalarType &cost, std::vector<std::vector<ScalarType>> &inputGrad, std::vector<std::vector<ScalarType>> &outputGrad)
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

template<class ScalarType>
void word2VecSGDWrapper(std::function<void(const std::string &,
                                                unsigned int,
                                                const std::vector<std::string> &,
                                                const std::map<std::string, unsigned int> &,
                                                const std::vector<std::vector<ScalarType>> &,
                                                const std::vector<std::vector<ScalarType>> &,
                                                std::function<void(const std::vector<ScalarType> &, unsigned int, const std::vector<std::vector<ScalarType>> &, ScalarType &, std::vector<ScalarType> &, std::vector<std::vector<ScalarType>> &)> ,
                                                ScalarType &, std::vector<std::vector<ScalarType>> &, std::vector<std::vector<ScalarType>> &)> word2VecModel, const std::map<std::string, unsigned int> &tokens, const std::vector<std::vector<ScalarType>> &inputWordVectors, const std::vector<std::vector<ScalarType>> &outputWordVectors, Word2VecDataset &dataset, unsigned int C, std::function<void(const std::vector<ScalarType> &, unsigned int, const std::vector<std::vector<ScalarType>> &, ScalarType &, std::vector<ScalarType> &, std::vector<std::vector<ScalarType>> &)> word2vecCostAndGradient, ScalarType &cost, std::vector<std::vector<ScalarType>> &inGrad, std::vector<std::vector<ScalarType>> &outGrad)
{
    const int batchsize = 250;
    cost = 0.0;
    inGrad.resize(inputWordVectors.size());
    for(int i = 0;i<inGrad.size();++i)
    {
        inGrad[i].resize(inputWordVectors[0].size(), 0.0);
    }
    outGrad.resize(outputWordVectors.size());
    for(int i =0;i<outGrad.size();++i)
    {
        outGrad[i].resize(outputWordVectors[0].size(), 0.0);
    }

    for(int i =0; i<batchsize;++i)
    {
        std::string centerWord;
        std::vector<std::string> context;
        unsigned int C1 = rand() % C + 1;
        dataset.getRandomContext(C1, centerWord, context);

        ScalarType c;
        std::vector<std::vector<ScalarType>> gin;
        std::vector<std::vector<ScalarType>> gout;

        word2VecModel(centerWord, C, context, tokens, inputWordVectors, outputWordVectors, word2vecCostAndGradient,  c, gin, gout);
        cost += c / batchsize;

        for(int h = 0;h<inGrad.size();++h)
        {
            for(int e=0;e<inGrad[0].size();++e)
            {
                inGrad[h][e] += gin[h][e] / batchsize;
                outGrad[h][e] += gout[h][e] / batchsize;

            }
        }
    }
}

template<class ScalarType>
void normalizeRows(std::vector<std::vector<ScalarType>> &inGrad,std::vector<std::vector<ScalarType>> &outGrad)
{
    for(int i = 0;i<inGrad.size();++i)
    {
        ScalarType lenIn = 0.0;
        ScalarType lenOut = 0.0;
        for(int e=0;e<inGrad[0].size();++e)
        {
            lenIn += inGrad[i][e]*inGrad[i][e];
            lenOut += outGrad[i][e]*outGrad[i][e];
        }

        lenIn = std::sqrt(lenIn);
        lenOut = std::sqrt(lenOut);

        for(int e=0;e<inGrad[0].size();++e)
        {
            inGrad[i][e] /= lenIn;
            outGrad[i][e] /= lenOut;
        }
    }
}

template<class ScalarType>
void word2VecSGD(std::vector<std::vector<ScalarType>> &inGrad0, std::vector<std::vector<ScalarType>> &outGrad0, Word2VecDataset &dataset, ScalarType step, unsigned int iterations)
{
    // Anneal learning rate every several iterations
    const unsigned int ANNEAL_EVERY = 20000;

    std::vector<std::vector<ScalarType>> inGrad = inGrad0;
    std::vector<std::vector<ScalarType>> outGrad = outGrad0;

    const std::map<std::string, unsigned int>  &tokens = dataset.tokens();

    normalizeRows(inGrad, outGrad);
    for(int i = 1; i< iterations+1; ++i)
    {
        ScalarType cost;
        std::vector<std::vector<ScalarType>> newInGrad;
        std::vector<std::vector<ScalarType>> newOutGrad;

        word2VecSGDWrapper<double>(skipgram<ScalarType>, tokens, inGrad, outGrad, dataset, 5, softmaxCostAndGradient<ScalarType>, cost, newInGrad, newOutGrad);

        for(int h = 0;h<inGrad.size();++h)
        {
            for(int e=0;e<inGrad[0].size();++e)
            {
                inGrad[h][e] -= step * newInGrad[h][e];
                outGrad[h][e] -= step * newOutGrad[h][e];
            }
        }

        normalizeRows(inGrad, outGrad);

        qDebug() << "iteration:" << i << "cost" << cost;

        if (i % ANNEAL_EVERY == 0)
        {
            step *= 0.5;
        }
    }
}
#endif // WORD2VECMODELS

