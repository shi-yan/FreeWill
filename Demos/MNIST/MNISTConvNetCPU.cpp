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

void MNIST::trainConvolutionalModel()
{
    qDebug() << "================== CPU Convolution network ==========================";

    const unsigned int featureMapSize = 20;
    const unsigned int batchSize = 10;

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, float> image({1,28,28,batchSize});
    image.init();

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, unsigned int> label({1, batchSize});
    label.init();

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, float> featureMap({1,5,5, featureMapSize});
    featureMap.init();
    featureMap.randomize();

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, float> bias({featureMapSize});
    bias.init();
    bias.randomize();

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, float> convOutput({featureMapSize,24,24,batchSize});
    convOutput.init();

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, float> poolingOutput({featureMapSize,12,12,batchSize});
    poolingOutput.init();

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, unsigned int> poolingSwitchX({featureMapSize,12,12,batchSize});
    poolingSwitchX.init();

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, unsigned int> poolingSwitchY({featureMapSize,12,12,batchSize});
    poolingSwitchY.init();

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, float> fullyConnected1Weight({100, featureMapSize*12*12});
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

    FreeWill::Convolution<FreeWill::DeviceType::CPU_NAIVE, float> convolution;
    convolution.setInputParameter("Input", &image);
    convolution.setInputParameter("FeatureMap", &featureMap);
    convolution.setInputParameter("Bias", &bias);
    convolution.setOutputParameter("Output", &convOutput);
    VERIFY_INIT(convolution.init());

    FreeWill::Activation<FreeWill::ActivationMode::SIGMOID, FreeWill::DeviceType::CPU_NAIVE, float> convSigmoid;
    convSigmoid.setInputParameter("Input", &convOutput);
    convSigmoid.setOutputParameter("Output", &convOutput);
    VERIFY_INIT(convSigmoid.init());

    FreeWill::MaxPooling<FreeWill::DeviceType::CPU_NAIVE, float> maxPooling;
    maxPooling.setInputParameter("Input", &convOutput);
    maxPooling.setOutputParameter("Output", &poolingOutput);
    maxPooling.setOutputParameter("SwitchX", &poolingSwitchX);
    maxPooling.setOutputParameter("SwitchY", &poolingSwitchY);
    VERIFY_INIT(maxPooling.init());

    poolingOutput.reshape({featureMapSize*12*12, batchSize});

    FreeWill::DotProductWithBias<FreeWill::DeviceType::CPU_NAIVE, float> fullyConnected1;
    fullyConnected1.setInputParameter("Input", &poolingOutput);
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
    dotProductWithBias1Derivative.setInputParameter("InputActivation", &poolingOutput);
    dotProductWithBias1Derivative.setInputParameter("OutputDelta", &fullyConnected1OutputGrad);
    dotProductWithBias1Derivative.setInputParameter("Weight", &fullyConnected1Weight);

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, float> poolingOutputGrad({featureMapSize*12*12,batchSize});
    poolingOutputGrad.init();

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, float> fullyConnected1WeightGrad({100, featureMapSize*12*12});
    fullyConnected1WeightGrad.init();

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, float> fullyConnected1BiasGrad({100});
    fullyConnected1BiasGrad.init();

    dotProductWithBias1Derivative.setOutputParameter("InputDelta", &poolingOutputGrad);
    dotProductWithBias1Derivative.setOutputParameter("BiasGrad", &fullyConnected1BiasGrad);
    dotProductWithBias1Derivative.setOutputParameter("WeightGrad", &fullyConnected1WeightGrad);

    VERIFY_INIT(dotProductWithBias1Derivative.init());

    poolingOutputGrad.reshape({featureMapSize,12,12,batchSize});

    FreeWill::MaxPoolingDerivative<FreeWill::DeviceType::CPU_NAIVE, float> maxPoolingDerivative;
    maxPoolingDerivative.setInputParameter("OutputGrad", &poolingOutputGrad);
    maxPoolingDerivative.setInputParameter("SwitchX", &poolingSwitchX);
    maxPoolingDerivative.setInputParameter("SwitchY", &poolingSwitchY);

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, float> convOutputGrad({featureMapSize,24,24,batchSize});
    convOutputGrad.init();

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, float> convBiasGrad({featureMapSize});
    convBiasGrad.init();

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, float> convFeatureMapGrad({1,5,5,featureMapSize});
    convFeatureMapGrad.init();

    maxPoolingDerivative.setOutputParameter("InputGrad", &convOutputGrad);
    VERIFY_INIT(maxPoolingDerivative.init());

    FreeWill::ActivationDerivative<FreeWill::ActivationMode::SIGMOID, FreeWill::DeviceType::CPU_NAIVE, float> convSigmoidDerivative;
    convSigmoidDerivative.setInputParameter("Output", &convOutput);
    convSigmoidDerivative.setInputParameter("OutputDelta", &convOutputGrad);
    convSigmoidDerivative.setOutputParameter("InputDelta", &convOutputGrad);
    VERIFY_INIT(convSigmoidDerivative.init());

    FreeWill::ConvolutionDerivative<FreeWill::DeviceType::CPU_NAIVE, float> convDerivative;
    convDerivative.setInputParameter("PrevActivation", &image);
    convDerivative.setInputParameter("FeatureMap", &featureMap);
    convDerivative.setInputParameter("OutputGrad", &convOutputGrad);

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, float> inputGrad({1,28,28,batchSize});
    inputGrad.init();

    convDerivative.setOutputParameter("FeatureMapGrad", &convFeatureMapGrad);
    convDerivative.setOutputParameter("BiasGrad", &convBiasGrad);
    convDerivative.setOutputParameter("InputGrad", &inputGrad);

    VERIFY_INIT(convDerivative.init());

    FreeWill::ElementwiseAdd<FreeWill::DeviceType::CPU_NAIVE, float> updateConvWeight;
    updateConvWeight.setInputParameter("OperandA", &featureMap);
    updateConvWeight.setInputParameter("OperandB", &convFeatureMapGrad);
    updateConvWeight.setOutputParameter("Result", &featureMap);

    VERIFY_INIT(updateConvWeight.init());

    FreeWill::ElementwiseAdd<FreeWill::DeviceType::CPU_NAIVE, float> updateConvBias;

    updateConvBias.setInputParameter("OperandA", &bias);
    updateConvBias.setInputParameter("OperandB", &convBiasGrad);
    updateConvBias.setOutputParameter("Result", &bias);

    VERIFY_INIT(updateConvBias.init());

    FreeWill::ElementwiseAdd<FreeWill::DeviceType::CPU_NAIVE, float> updateFullyConnected1Weight;
    updateFullyConnected1Weight.setInputParameter("OperandA", &fullyConnected1Weight);
    updateFullyConnected1Weight.setInputParameter("OperandB", &fullyConnected1WeightGrad);
    updateFullyConnected1Weight.setOutputParameter("Result", &fullyConnected1Weight);

    VERIFY_INIT(updateFullyConnected1Weight.init());

    FreeWill::ElementwiseAdd<FreeWill::DeviceType::CPU_NAIVE, float> updateFullyConnected2Weight;
    updateFullyConnected2Weight.setInputParameter("OperandA", &fullyConnected2Weight);
    updateFullyConnected2Weight.setInputParameter("OperandB", &fullyConnected2WeightGrad);
    updateFullyConnected2Weight.setOutputParameter("Result", &fullyConnected2Weight);

    VERIFY_INIT(updateFullyConnected2Weight.init());

    FreeWill::ElementwiseAdd<FreeWill::DeviceType::CPU_NAIVE, float> updateFullyConnected1Bias;
    updateFullyConnected1Bias.setInputParameter("OperandA", &fullyConnected1Bias);
    updateFullyConnected1Bias.setInputParameter("OperandB", &fullyConnected1BiasGrad);
    updateFullyConnected1Bias.setOutputParameter("Result", &fullyConnected1Bias);

    VERIFY_INIT(updateFullyConnected1Bias.init());

    FreeWill::ElementwiseAdd<FreeWill::DeviceType::CPU_NAIVE, float> updateFullyConnected2Bias;
    updateFullyConnected2Bias.setInputParameter("OperandA", &fullyConnected2Bias);
    updateFullyConnected2Bias.setInputParameter("OperandB", &fullyConnected2BiasGrad);
    updateFullyConnected2Bias.setOutputParameter("Result", &fullyConnected2Bias);

    VERIFY_INIT(updateFullyConnected2Bias.init());


    //closeFileDump();

    float learningRate = 0.01;
    float overallCost = 0.0;
    float accuracy = 0.0;

    for(unsigned int e = 1;e<=60;++e)
    {
        openTrainData();
        const int testInterval = numOfImage/batchSize;

        for(unsigned int i = 1;i<=numOfImage/batchSize;++i)
        {
            //openData();
            loadOneTrainData(image, label, batchSize);

            //forward
            convolution.evaluate();

            convSigmoid.evaluate();
            poolingOutput.reshape({featureMapSize,12,12,batchSize});
            maxPooling.evaluate();
            poolingOutput.reshape({featureMapSize*12*12,batchSize});
            fullyConnected1.evaluate();
            sigmoid1.evaluate();
            fullyConnected1Output.copyFromDeviceToHost();
            fullyConnected2.evaluate();
            softmaxLogLoss.evaluate();

            for(unsigned int c = 0;c<batchSize;++c)
            {
                overallCost += cost[c];
            }
            softmaxLogLossDerivative.evaluate();
            dotProductWithBias2Derivative.evaluate();
            sigmoidDerivative.evaluate();
            poolingOutputGrad.reshape({featureMapSize*12*12,batchSize});
            dotProductWithBias1Derivative.evaluate();
            poolingOutputGrad.reshape({featureMapSize,12,12,batchSize});
            maxPoolingDerivative.evaluate();
            convSigmoidDerivative.evaluate();
            convDerivative.evaluate();

            qDebug() << e << i<< "cost" << overallCost / (float) batchSize << learningRate << accuracy;
            emit updateCost(overallCost / (float) batchSize);
            overallCost = 0.0;

            updateConvWeight.setRate(-learningRate);
            updateConvWeight.evaluate();
            updateConvBias.setRate(-learningRate);
            updateConvBias.evaluate();
            updateFullyConnected1Weight.setRate(-learningRate);
            updateFullyConnected1Weight.evaluate();
            updateFullyConnected1Bias.setRate(-learningRate);
            updateFullyConnected1Bias.evaluate();
            updateFullyConnected2Weight.setRate(-learningRate);
            updateFullyConnected2Weight.evaluate();
            updateFullyConnected2Bias.setRate(-learningRate);
            updateFullyConnected2Bias.evaluate();

            if (i%testInterval == 0)
            {
                learningRate *= 0.95;
            }

            if (i % testInterval == 0)
            {
                unsigned int correct = 0;

                openTestData();

                for (unsigned int v = 0;v<numOfTestImage/batchSize; ++v)
                {
                    convOutput.clear();
                    poolingOutput.clear();
                    poolingSwitchX.clear();
                    poolingSwitchY.clear();
                    fullyConnected1Output.clear();
                    fullyConnected2Output.clear();
                    softmaxOutput.clear();
                    cost.clear();
                    softmaxGrad.clear();

                    loadOneTestData(image, label, batchSize);

                    convolution.evaluate();
                    convSigmoid.evaluate();
                    poolingOutput.reshape({featureMapSize,12,12,10});
                    maxPooling.evaluate();
                    poolingOutput.reshape({featureMapSize*12*12,10});
                    fullyConnected1.evaluate();
                    sigmoid1.evaluate();
                    fullyConnected2.evaluate();
                    softmaxLogLoss.evaluate();




                    for(unsigned int b =0;b<batchSize;++b)
                    {
                        unsigned int maxIndex = 0;
                        float maxValue = softmaxOutput[b*softmaxOutput.shape()[0]];
                        for(unsigned int e = 1; e < softmaxOutput.shape()[0]; ++e)
                        {
                            if (maxValue < softmaxOutput[b*softmaxOutput.shape()[0] + e])
                            {
                                maxValue = softmaxOutput[b*softmaxOutput.shape()[0] + e];
                                maxIndex = e;
                            }
                        }

                        if ((float) maxIndex == label[b])
                        {
                            correct ++;
                        }
                    }

                 }

                 qDebug() << "Accuracy" << (accuracy = (float) correct / (float) numOfTestImage);

                 closeTestData();
            }

            convOutput.clear();
            poolingOutput.clear();
            poolingSwitchX.clear();
            poolingSwitchY.clear();
            fullyConnected1Output.clear();
            fullyConnected2Output.clear();
            softmaxOutput.clear();
            cost.clear();
            softmaxGrad.clear();
            fullyConnected1OutputGrad.clear();
            fullyConnected2WeightGrad.clear();
            fullyConnected1WeightGrad.clear();
            fullyConnected1BiasGrad.clear();
            fullyConnected2BiasGrad.clear();
            poolingOutputGrad.clear();
            convOutputGrad.clear();
            convBiasGrad.clear();
            convFeatureMapGrad.clear();
            inputGrad.clear();

            emit updateProgress(i*batchSize / (float)(numOfImage), ((e-1)*numOfImage + i*batchSize) / (60.0f*numOfImage));
        }
        closeTrainData();
    }
}
