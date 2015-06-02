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

    VectorType result(VectorType input)
    {
        VectorType previousNeurons;

        previousNeurons = input;

        foreach(std::vector<VectorType> &weightsForOneLayer, weights)
        {
            VectorType resultLayer;
            foreach(VectorType &weightsForOneNeuron, weightsForOneLayer)
            {
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

                    resultLayer.push_back(result);
                }
                else
                {
                    qDebug() << "layer size doesn't add up" << (previousNeurons.size() + 1) << weightsForOneNeuron.size();
                }
            }
            previousNeurons = resultLayer;
        }

        return previousNeurons;
    }

    ScalarType error(VectorType result, VectorType expectedAnswer)
    {
        if (result.size() == expectedAnswer.size())
        {
            ScalarType sum = 0.0;
            for(int i = 0; i< result.size() ;++i)
            {
                sum+=(expectedAnswer - result)^2;
            }
            sum /= result.size();
            return sum;
        }
        else
        {
            qDebug() << "result size and expected answer size doesn't match" << result.size() << expectedAnswer.size();
            return 0;
        }
    }
    void save(QString &filePath);
    void load(QString &filePath);
    void backpropagate(ScalarType error);

private:
    std::vector<std::vector<VectorType>> weights;
    //std::vector<VectorType> neurons;
};

#endif // NEURALNETWORK_H
