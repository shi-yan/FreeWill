#include <QCoreApplication>
#include "GradientCheck.h"
#include <QDebug>
#include <vector>
#include "NeuralNetwork.h"
#include "GradientCheck.h"
#include "Word2VecModels.h"
#include "StanfordSentimentDataset.h"

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    //testGradientCheck();

    //testNeuralNetwork();
    StanfordSentimentDataset dataset("stanfordSentimentTreebank", 10000);

    unsigned int nToken = dataset.tokens().size();

    std::vector<std::vector<double>> inGrad;
    std::vector<std::vector<double>> outGrad;

    unsigned int previous = 1000;

    if (!previous)
    {
        inGrad.resize(nToken);
        outGrad.resize(nToken);

        for(int i = 0;i<nToken;++i)
        {
            inGrad[i].resize(10);
            outGrad[i].resize(10);

            for(int e = 0;e<10;++e)
            {
                inGrad[i][e] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
                outGrad[i][e] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            }
        }

    }
    else
    {
        load(previous, inGrad, outGrad);
    }

    word2VecSGD<double>(previous, inGrad, outGrad, dataset, 0.5, 400000);

    return a.exec();
}
