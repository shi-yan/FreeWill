#include "Tensor/Tensor.h"
#include "Operator/MaxPooling.h"
#include "Operator/MaxPoolingDerivative.h"
#include <cstdio>
#include "Operator/Convolution.h"
#include "Operator/ConvolutionDerivative.h"
#include "Operator/SoftmaxLogLoss.h"
#include "Operator/SoftmaxLogLossDerivative.h"
#include "Operator/DotProductWithBias.h"
#include "Operator/DotProductWithBiasDerivative.h"
#include "Operator/Activation.h"
#include "Operator/ActivationDerivative.h"
#include "Operator/ElementwiseProduct.h"
#include <QDebug>
#include "Operator/ElementwiseAdd.h"
#include "MNIST.h"

void MNIST::trainFullyConnectedModel()
{
    qDebug() << "================== CPU Fully Connected network ==========================";

    unsigned int batchSize = 10;

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, float> image({28*28,batchSize});
    image.init();

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, unsigned int> label({1, batchSize});
    label.init();

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, float> fullyConnected1Weight({100, 28*28});
    fullyConnected1Weight.init();
    fullyConnected1Weight.randomize();

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, float> fullyConnected1Bias({100});
    fullyConnected1Bias.init();
    fullyConnected1Bias.randomize();

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, float> fullyConnected1Output({100, batchSize});
    fullyConnected1Output.init();

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, float> fullyConnected2Weight({10, 100});
    fullyConnected2Weight.init();
    fullyConnected2Weight.randomize();

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, float> fullyConnected2Bias({10});
    fullyConnected2Bias.init();
    fullyConnected2Bias.randomize();

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, float> fullyConnected2Output({10, batchSize});
    fullyConnected2Output.init();

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, float> softmaxOutput({10,batchSize});
    softmaxOutput.init();

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, float> cost({1, batchSize});
    cost.init();

    FreeWill::DotProductWithBias<FreeWill::DeviceType::CPU_NAIVE, float> fullyConnected1;
    fullyConnected1.setInputParameter("Input", &image);
    fullyConnected1.setInputParameter("Weight", &fullyConnected1Weight);
    fullyConnected1.setInputParameter("Bias", &fullyConnected1Bias);
    fullyConnected1.setOutputParameter("Output", &fullyConnected1Output);
    VERIFY_INIT(fullyConnected1.init());

    FreeWill::Activation<FreeWill::ActivationMode::SIGMOID, FreeWill::DeviceType::CPU_NAIVE, float> sigmoid1;
    sigmoid1.setInputParameter("Input", &fullyConnected1Output);
    sigmoid1.setOutputParameter("Output", &fullyConnected1Output);
    VERIFY_INIT(sigmoid1.init());

    FreeWill::DotProductWithBias<FreeWill::DeviceType::CPU_NAIVE, float> fullyConnected2;
    fullyConnected2.setInputParameter("Input", &fullyConnected1Output);
    fullyConnected2.setInputParameter("Weight", &fullyConnected2Weight);
    fullyConnected2.setInputParameter("Bias", &fullyConnected2Bias);
    fullyConnected2.setOutputParameter("Output", &fullyConnected2Output);
    VERIFY_INIT(fullyConnected2.init());

    FreeWill::SoftmaxLogLoss<FreeWill::DeviceType::CPU_NAIVE, float> softmaxLogLoss;
    softmaxLogLoss.setInputParameter("Input", &fullyConnected2Output);
    softmaxLogLoss.setInputParameter("Label", &label);
    softmaxLogLoss.setOutputParameter("Output", &softmaxOutput);
    softmaxLogLoss.setOutputParameter("Cost", &cost);
    VERIFY_INIT(softmaxLogLoss.init());


    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, float> softmaxGrad({10,batchSize});
    softmaxGrad.init();

    FreeWill::SoftmaxLogLossDerivative<FreeWill::DeviceType::CPU_NAIVE, float> softmaxLogLossDerivative;
    softmaxLogLossDerivative.setInputParameter("Output", &softmaxOutput);
    softmaxLogLossDerivative.setInputParameter("Label", &label);
    softmaxLogLossDerivative.setOutputParameter("InputGrad", &softmaxGrad);
    VERIFY_INIT(softmaxLogLossDerivative.init());

    FreeWill::DotProductWithBiasDerivative<FreeWill::DeviceType::CPU_NAIVE, float> dotProductWithBias2Derivative;
    dotProductWithBias2Derivative.setInputParameter("InputActivation", &fullyConnected1Output);
    dotProductWithBias2Derivative.setInputParameter("OutputDelta", &softmaxGrad);
    dotProductWithBias2Derivative.setInputParameter("Weight", &fullyConnected2Weight);

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, float> fullyConnected1OutputGrad({100,batchSize});
    fullyConnected1OutputGrad.init();

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, float> fullyConnected2WeightGrad({10,100});
    fullyConnected2WeightGrad.init();

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, float> fullyConnected2BiasGrad({10});
    fullyConnected2BiasGrad.init();

    dotProductWithBias2Derivative.setOutputParameter("InputDelta", &fullyConnected1OutputGrad);
    dotProductWithBias2Derivative.setOutputParameter("BiasGrad", &fullyConnected2BiasGrad);
    dotProductWithBias2Derivative.setOutputParameter("WeightGrad", &fullyConnected2WeightGrad);

    VERIFY_INIT(dotProductWithBias2Derivative.init());

    FreeWill::ActivationDerivative<FreeWill::ActivationMode::SIGMOID, FreeWill::DeviceType::CPU_NAIVE, float> sigmoidDerivative;
    sigmoidDerivative.setInputParameter("Output", &fullyConnected1Output);
    sigmoidDerivative.setInputParameter("OutputDelta", &fullyConnected1OutputGrad);
    sigmoidDerivative.setOutputParameter("InputDelta", &fullyConnected1OutputGrad);

    VERIFY_INIT(sigmoidDerivative.init());

    FreeWill::DotProductWithBiasDerivative<FreeWill::DeviceType::CPU_NAIVE, float> dotProductWithBias1Derivative;
    dotProductWithBias1Derivative.setInputParameter("InputActivation", &image);
    dotProductWithBias1Derivative.setInputParameter("OutputDelta", &fullyConnected1OutputGrad);
    dotProductWithBias1Derivative.setInputParameter("Weight", &fullyConnected1Weight);

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, float> fullyConnected1WeightGrad({100, 28*28});
    fullyConnected1WeightGrad.init();

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, float> imageGrad({28*28, batchSize});
    imageGrad.init();

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, float> fullyConnected1BiasGrad({100});
    fullyConnected1BiasGrad.init();

    dotProductWithBias1Derivative.setOutputParameter("InputDelta", &imageGrad);
    dotProductWithBias1Derivative.setOutputParameter("BiasGrad", &fullyConnected1BiasGrad);
    dotProductWithBias1Derivative.setOutputParameter("WeightGrad", &fullyConnected1WeightGrad);

    VERIFY_INIT(dotProductWithBias1Derivative.init());

    FreeWill::ElementwiseAdd<FreeWill::DeviceType::CPU_NAIVE, float> updateFullyConnected1Weight;
    updateFullyConnected1Weight.setInputParameter("OperandA", &fullyConnected1Weight);
    updateFullyConnected1Weight.setInputParameter("OperandB", &fullyConnected1WeightGrad);
    updateFullyConnected1Weight.setOutputParameter("Result", &fullyConnected1Weight);

    VERIFY_INIT(updateFullyConnected1Weight.init());

    FreeWill::ElementwiseAdd<FreeWill::DeviceType::CPU_NAIVE, float> updateFullyConnected1Bias;
    updateFullyConnected1Bias.setInputParameter("OperandA", &fullyConnected1Bias);
    updateFullyConnected1Bias.setInputParameter("OperandB", &fullyConnected1BiasGrad);
    updateFullyConnected1Bias.setOutputParameter("Result", &fullyConnected1Bias);

    VERIFY_INIT(updateFullyConnected1Bias.init());

    FreeWill::ElementwiseAdd<FreeWill::DeviceType::CPU_NAIVE, float> updateFullyConnected2Weight;
    updateFullyConnected2Weight.setInputParameter("OperandA", &fullyConnected2Weight);
    updateFullyConnected2Weight.setInputParameter("OperandB", &fullyConnected2WeightGrad);
    updateFullyConnected2Weight.setOutputParameter("Result", &fullyConnected2Weight);

    VERIFY_INIT(updateFullyConnected2Weight.init());

    FreeWill::ElementwiseAdd<FreeWill::DeviceType::CPU_NAIVE, float> updateFullyConnected2Bias;
    updateFullyConnected2Bias.setInputParameter("OperandA", &fullyConnected2Bias);
    updateFullyConnected2Bias.setInputParameter("OperandB", &fullyConnected2BiasGrad);
    updateFullyConnected2Bias.setOutputParameter("Result", &fullyConnected2Bias);

    VERIFY_INIT(updateFullyConnected2Bias.init());

    float learningRate = 0.01;

    float overallCost = 0.0;
    const int testInterval = 2000;

    for(unsigned int e = 1; e<=60; ++e)
    {
        openTrainData();

        for(unsigned int i = 1; i<=numOfImage/batchSize; ++i)
        {
            loadOneTrainData(image, label, batchSize);

            fullyConnected1.evaluate();
            sigmoid1.evaluate();
            //ReLu1.evaluate();
            fullyConnected2.evaluate();
            softmaxLogLoss.evaluate();

            for(unsigned int c = 0; c<batchSize; ++c)
            {
                overallCost += cost[c];
            }

            qDebug() << e << i<< "cost" << overallCost / (float) batchSize << learningRate;

            fullyConnected1BiasGrad.clear();
            fullyConnected1WeightGrad.clear();
            fullyConnected2BiasGrad.clear();
            fullyConnected2WeightGrad.clear();

            //backward
            softmaxLogLossDerivative.evaluate();
            dotProductWithBias2Derivative.evaluate();
            sigmoidDerivative.evaluate();
            dotProductWithBias1Derivative.evaluate();


            emit updateCost(overallCost / (float) batchSize);
            overallCost = 0.0;

            //update weight
            updateFullyConnected1Weight.setRate(-learningRate);
            updateFullyConnected1Weight.evaluate();
            updateFullyConnected1Bias.setRate(-learningRate);
            updateFullyConnected1Bias.evaluate();

            updateFullyConnected2Weight.setRate(-learningRate);
            updateFullyConnected2Weight.evaluate();

            updateFullyConnected2Bias.setRate(-learningRate);
            updateFullyConnected2Bias.evaluate();

            if (i%30000 == 0)
            {
               learningRate *= 0.9;
            }


            if (i % testInterval == 0)
            {
                unsigned int correct = 0;

                openTestData();

                for (unsigned int v = 0;v<numOfTestImage/batchSize; ++v)
                {
                    fullyConnected1Output.clear();
                    fullyConnected2Output.clear();
                    softmaxOutput.clear();
                    cost.clear();
                    softmaxGrad.clear();

                    loadOneTestData(image, label,batchSize);

                    //forward
                    fullyConnected1.evaluate();
                    sigmoid1.evaluate();
                    //ReLu1.evaluate();
                    fullyConnected2.evaluate();
                    softmaxLogLoss.evaluate();

                    for (unsigned int b = 0;b<batchSize;++b)
                    {
                        unsigned int maxIndex = 0;
                        float maxValue = softmaxOutput[b * softmaxOutput.shape()[0]];
                        for(unsigned int e = 1; e < softmaxOutput.shape()[0]; ++e)
                        {
                            if (maxValue < softmaxOutput[b* softmaxOutput.shape()[0] + e])
                            {
                                maxValue = softmaxOutput[b* softmaxOutput.shape()[0] + e];
                                maxIndex = e;
                            }
                        }

                        if ((float) maxIndex == label[b])
                        {
                            correct ++;
                        }
                    }


                }
                qDebug() << "Accuracy" << (float) correct / (float) numOfTestImage;

                closeTestData();

                //clean up

                fullyConnected1Output.clear();
                fullyConnected2Output.clear();
                softmaxOutput.clear();
                cost.clear();
                softmaxGrad.clear();
                fullyConnected1OutputGrad.clear();
                fullyConnected2BiasGrad.clear();
                fullyConnected2WeightGrad.clear();
                fullyConnected1BiasGrad.clear();
                fullyConnected1WeightGrad.clear();
                imageGrad.clear();

                //closeData();

            }

            emit updateProgress(i*batchSize / (float)(numOfImage), ((e-1)*numOfImage + i*batchSize) / (60.0f*numOfImage));

            /*if (e % 10000 == 0)
            {
                learningRate *= 0.8;
            }*/

        }
        closeTrainData();
        //closeData();
    }
}
