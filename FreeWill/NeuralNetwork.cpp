#include "NeuralNetwork.h"
#include "ActivationFunctions.h"
#include "CostFunctions.h"

void testNeuralNetwork()
{
    NeuralNetwork<float> network;
    std::vector<unsigned int> layerCounts;
    layerCounts.push_back(5);
    network.init(10,10,layerCounts, sigmoid<float>, sigmoid<float>, crossEntropy<float>, derivativeCrossEntropySigmoid<float>);
}
