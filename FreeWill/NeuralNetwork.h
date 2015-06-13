#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>
#include <QString>
#include <time.h>
#include "Sigmoid.h"

template<class ScalarType, class VectorType>
class NeuralNetwork
{
public:
    NeuralNetwork(){}
    ~NeuralNetwork(){}

    void init(int inputSize, std::vector<int> neuronCounts)
    {
        int previousLayerSize = inputSize + 1;

        foreach(int count, neuronCounts)
        {
            VectorType neuronLayer;
            neuronLayer.resize(count + 1, 0.0);
            neurons.push_back(neuronLayer);

            std::vector<VectorType> weightLayer;
            for(int i = 0; i < (count + 1); ++i)
            {
                VectorType weightsForNeuron;
                weightsForNeuron.resize(previousLayerSize);
                weightLayer.push_back(weightsForNeuron);
            }
            weights.push_back(weightLayer);
            previousLayerSize = count + 1;
        }
    }

    void randomWeights()
    {
        srand(time(null));

        foreach(std::vector<VectorType> &weightsForOneLayer, weights)
        {
            foreach(VectorType &weightsForOneNeuron, weightsForOneLayer)
            {
                for(int i = 0; i < weightsForOneNeuron.size(); ++i)
                {
                    weightsForOneNeuron[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
                }
            }
        }
    }

    ScalarType train(std::vector<VectorType> miniBatch, std::vector<VectorType> expectedAnswer, ScalarType &cost)
    {
        if (miniBatch.size() != expectedAnswer)
        {
            qDebug() << "inputs should have the same size as the outputs";
            return 0;
        }
        //batch     //layer     //neuron
        std::vector<std::vector<std::vector<ScalarType>>> activations;

        activations.resize(miniBatch.size());

        for(int i = 0; i< miniBatch.size(); ++i)
        {
            VectorType const &previousNeurons = miniBatch[i];

            activations[i].resize(weights.size());
            for(int l = 0; l< weights.size(); ++l)
            {
                std::vector<VectorType> &weightsForOneLayer = weights[l];
                activations[i][l].resize(weightsForOneLayer.size()-1, 0);

                for(int n = 0; n < weightsForOneLayer.size()-1; ++n)
                {
                    VectorType &weightsForOneNeuron = weightsForOneLayer[n];

                    if(previousNeurons.size() + 1 == weightsForOneNeuron.size())
                    {
                        ScalarType result = 0.0;
                        int i = 0;
                        for(i = 0; i < previousNeurons.size(); ++i)
                        {
                            result += previousNeurons[i] * weightsForOneNeuron[i];
                        }
                        result += weightsForOneNeuron[i];
                        result = sigmoid<ScalarType>(result);

                        activations[i][l][n] = result;
                    }
                    else
                    {
                        qDebug() << "layer size doesn't add up" << (previousNeurons.size() + 1) << weightsForOneNeuron.size();
                        return 0;
                    }
                }
                previousNeurons = resultLayer;
            }
        }

        ScalarType crossEntropy = 0.0;
        for(int i = 0;i<activations.size();++i)
        {
            int finalLayer = activations[i].size() - 1;
            int neuronCount = activations[i][finalLayer].size();
            for(int e = 0; e< neuronCount;++e)
            {
                crossEntropy += expectedAnswer[i][e]*log(activations[i][finalLayer][e]) + (1- expectedAnswer[i][e])*log(1-activations[i][finalLayer][e]);
            }
        }

        crossEntropy *= -1.0/miniBatch.size();

        cost = crossEntropy;

        std::vector<std::vector<std::vector<ScalarType>>> delta;

        delta.resize(miniBatch.size());

        for(int i = 0; i< miniBatch.size();++i)
        {
            delta[i].resize(weights.size());
            delta[i][delta[i].size() -1].resize(weights[weights.size() -1].size() -1, 0);
            for(int e = 0; e< weights[weights.size() -1].size() -1;++e)
            {
                delta[i][delta[i].size() - 1][e] = activations[i][delta[i].size()-1][e] - expectedAnswer[i][e];
            }
        }

        std::vector<std::vector<ScalarType>> bias;

        bias.resize(weights.size());


        for(int i = 0; i<miniBatch.size(); ++i)
        {

            bias[weights.size() - 1].resize(weights[weights.size() -1].size() -1);

            for(int e = 0; e< weights[weights.size() -1].size() -1;++e)
            {
                bias[weights.size()-1][e] = delta[i][delta[i].size()-1][e];
            }
        }

    }


private:
    std::vector<std::vector<VectorType>> weights;

};

#endif // NEURALNETWORK_H
