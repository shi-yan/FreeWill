#include "endian.h"
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
#include <QHostAddress>
#include "MNIST.h"

MNIST::MNIST(WebsocketServer *websocketServer, bool usingConvolution)
    :QThread(),
    datafp(NULL),
    labelfp(NULL),
    numOfImage(0),
    numOfRow(0),
    numOfColumn(0),
    labelCount(0),
    testDatafp(NULL),
    testLabelfp(NULL),
    numOfTestImage(0),
    numOfTestRow(0),
    numOfTestColumn(0),
    labelTestCount(0),
    m_websocketServer(websocketServer),
    m_usingConvolution(usingConvolution)
{

}

MNIST::~MNIST()
{
}

void MNIST::openTrainData()
{
    datafp = fopen("train-images-idx3-ubyte","rb");
    labelfp = fopen("train-labels-idx1-ubyte","rb");

    unsigned int magicNum = 0;
    numOfImage = 0;
    numOfRow = 0;
    numOfColumn = 0;

    unsigned int magicNumLabel = 0;
    labelCount = 0;

    fread(&magicNumLabel, sizeof(unsigned int), 1, labelfp);
    fread(&labelCount, sizeof(unsigned int ),1, labelfp);

    magicNumLabel = be32toh(magicNumLabel);
    labelCount = be32toh(labelCount);

    fread(&magicNum, sizeof(unsigned int), 1, datafp);
    fread(&numOfImage, sizeof(unsigned int), 1, datafp);
    fread(&numOfRow, sizeof(unsigned int), 1, datafp);
    fread(&numOfColumn, sizeof(unsigned int), 1, datafp);

    magicNum = be32toh(magicNum);
    numOfImage = be32toh(numOfImage);
    numOfRow = be32toh(numOfRow);
    numOfColumn = be32toh(numOfColumn);
}

void MNIST::closeTrainData()
{
    fclose(datafp);
    fclose(labelfp);
}

void MNIST::openTestData()
{
    //t10k-images-idx3-ubyte  t10k-labels-idx1-ubyte
    testDatafp = fopen("t10k-images-idx3-ubyte","rb");
    testLabelfp = fopen("t10k-labels-idx1-ubyte","rb");

    unsigned int magicNum = 0;
    numOfTestImage = 0;
    numOfTestRow = 0;
    numOfTestColumn = 0;

    unsigned int magicNumLabel = 0;
    labelTestCount = 0;

    fread(&magicNumLabel, sizeof(unsigned int), 1, testLabelfp);
    fread(&labelTestCount, sizeof(unsigned int ),1, testLabelfp);

    magicNumLabel = be32toh(magicNumLabel);
    labelTestCount = be32toh(labelTestCount);

    fread(&magicNum, sizeof(unsigned int), 1, testDatafp);
    fread(&numOfTestImage, sizeof(unsigned int), 1, testDatafp);
    fread(&numOfTestRow, sizeof(unsigned int), 1, testDatafp);
    fread(&numOfTestColumn, sizeof(unsigned int), 1, testDatafp);

    magicNum = be32toh(magicNum);
    numOfTestImage = be32toh(numOfTestImage);
    numOfTestRow = be32toh(numOfTestRow);
    numOfTestColumn = be32toh(numOfTestColumn);
}

void MNIST::closeTestData()
{
    fclose(testDatafp);
    fclose(testLabelfp);
}

void MNIST::loadOneTrainData(FreeWill::Tensor<FreeWill::CPU, float> &image, FreeWill::Tensor<FreeWill::CPU, unsigned int> &label)
{ 
    for(unsigned int y = 0 ; y < numOfRow; ++y)
    {
        for(unsigned int x = 0;x< numOfColumn; ++x)
        {
            unsigned char pixel = 0;
            fread(&pixel, sizeof(unsigned char), 1, datafp);
            image[numOfColumn * y + x] = (float) pixel / 255.0f;
        }
    }
    unsigned char _label = 0;
    fread(&_label, sizeof(unsigned char), 1, labelfp);
    label[0] = _label;
}


void MNIST::loadOneTrainDataGPU(FreeWill::Tensor<FreeWill::GPU_CUDA, float> &image, FreeWill::Tensor<FreeWill::GPU_CUDA, unsigned int> &label)
{
    for(unsigned int y = 0 ; y < numOfRow; ++y)
    {
        for(unsigned int x = 0;x< numOfColumn; ++x)
        {
            unsigned char pixel = 0;
            fread(&pixel, sizeof(unsigned char), 1, datafp);
            image[numOfColumn * y + x] = (float) pixel / 255.0f;
        }
    }
    unsigned char _label = 0;
    fread(&_label, sizeof(unsigned char), 1, labelfp);
    label[0] = _label;

    label.copyFromHostToDevice();
    image.copyFromHostToDevice();
}

void MNIST::loadOneTestData(FreeWill::Tensor<FreeWill::CPU, float> &image, FreeWill::Tensor<FreeWill::CPU, unsigned int> &label)
{
    for(unsigned int y = 0 ; y < numOfTestRow; ++y)
    {
        for(unsigned int x = 0;x< numOfTestColumn; ++x)
        {
            unsigned char pixel = 0;
            fread(&pixel, sizeof(unsigned char), 1, testDatafp);
            image[numOfTestColumn * y + x] = (float) pixel / 255.0f;
        }
    }
    unsigned char _label = 0;
    fread(&_label, sizeof(unsigned char), 1, testLabelfp);
    label[0] = _label;
}

void MNIST::loadOneTestDataGPU(FreeWill::Tensor<FreeWill::GPU_CUDA, float> &image, FreeWill::Tensor<FreeWill::GPU_CUDA, unsigned int> &label)
{
    for(unsigned int y = 0 ; y < numOfTestRow; ++y)
    {
        for(unsigned int x = 0;x< numOfTestColumn; ++x)
        {
            unsigned char pixel = 0;
            fread(&pixel, sizeof(unsigned char), 1, testDatafp);
            image[numOfTestColumn * y + x] = (float) pixel / 255.0f;
        }
    }
    unsigned char _label = 0;
    fread(&_label, sizeof(unsigned char), 1, testLabelfp);
    label[0] = _label;

    image.copyFromHostToDevice();
    label.copyFromHostToDevice();
}


void MNIST::trainConvolutionalModel()
{

    const unsigned int featureMapSize = 20;

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> image({1,28,28,1});
    image.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, unsigned int> label({1});
    label.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> featureMap({1,5,5, featureMapSize});
    featureMap.init();
    featureMap.randomize();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> bias({featureMapSize});
    bias.init();
    bias.randomize();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> convOutput({featureMapSize,24,24,1});
    convOutput.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> poolingOutput({featureMapSize,12,12,1});
    poolingOutput.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, unsigned int> poolingSwitchX({featureMapSize,12,12,1});
    poolingSwitchX.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, unsigned int> poolingSwitchY({featureMapSize,12,12,1});
    poolingSwitchY.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> fullyConnected1Weight({100, featureMapSize*12*12});
    fullyConnected1Weight.init();
    fullyConnected1Weight.randomize();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> fullyConnected1Bias({100});
    fullyConnected1Bias.init();
    fullyConnected1Bias.randomize();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> fullyConnected1Output({100, 1});
    fullyConnected1Output.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> fullyConnected2Weight({10, 100});
    fullyConnected2Weight.init();
    fullyConnected2Weight.randomize();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> fullyConnected2Bias({10});
    fullyConnected2Bias.init();
    fullyConnected2Bias.randomize();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> fullyConnected2Output({10, 1});
    fullyConnected2Output.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> softmaxOutput({10,1});
    softmaxOutput.init();    

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> cost({1});
    cost.init();


    FreeWill::Convolution<FreeWill::CPU_NAIVE, float> convolution;
    convolution.setInputParameter("Input", &image);
    convolution.setInputParameter("FeatureMap", &featureMap);
    convolution.setInputParameter("Bias", &bias);
    convolution.setOutputParameter("Output", &convOutput);
    VERIFY_INIT(convolution.init());

    FreeWill::Activation<FreeWill::SIGMOID, FreeWill::CPU_NAIVE, float> convSigmoid;
    convSigmoid.setInputParameter("Input", &convOutput);
    convSigmoid.setOutputParameter("Output", &convOutput);
    VERIFY_INIT(convSigmoid.init());
    
    FreeWill::MaxPooling<FreeWill::CPU_NAIVE, float> maxPooling;
    maxPooling.setInputParameter("Input", &convOutput);
    maxPooling.setOutputParameter("Output", &poolingOutput);
    maxPooling.setOutputParameter("SwitchX", &poolingSwitchX);
    maxPooling.setOutputParameter("SwitchY", &poolingSwitchY);
    VERIFY_INIT(maxPooling.init());


    poolingOutput.reshape({featureMapSize*12*12, 1});

    FreeWill::DotProductWithBias<FreeWill::CPU_NAIVE, float> fullyConnected1;
    fullyConnected1.setInputParameter("Input", &poolingOutput);
    fullyConnected1.setInputParameter("Weight", &fullyConnected1Weight);
    fullyConnected1.setInputParameter("Bias", &fullyConnected1Bias);
    fullyConnected1.setOutputParameter("Output", &fullyConnected1Output);
    VERIFY_INIT(fullyConnected1.init());

    FreeWill::Activation<FreeWill::SIGMOID, FreeWill::CPU_NAIVE, float> sigmoid1;
    sigmoid1.setInputParameter("Input", &fullyConnected1Output);
    sigmoid1.setOutputParameter("Output", &fullyConnected1Output);
    VERIFY_INIT(sigmoid1.init());


    FreeWill::DotProductWithBias<FreeWill::CPU_NAIVE, float> fullyConnected2;
    fullyConnected2.setInputParameter("Input", &fullyConnected1Output);
    fullyConnected2.setInputParameter("Weight", &fullyConnected2Weight);
    fullyConnected2.setInputParameter("Bias", &fullyConnected2Bias);
    fullyConnected2.setOutputParameter("Output", &fullyConnected2Output);
    VERIFY_INIT(fullyConnected2.init());

    FreeWill::SoftmaxLogLoss<FreeWill::CPU_NAIVE, float> softmaxLogLoss;
    softmaxLogLoss.setInputParameter("Input", &fullyConnected2Output);
    softmaxLogLoss.setInputParameter("Label", &label);
    softmaxLogLoss.setOutputParameter("Output", &softmaxOutput);
    softmaxLogLoss.setOutputParameter("Cost", &cost);
    VERIFY_INIT(softmaxLogLoss.init());


    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> softmaxGrad({10,1});
    softmaxGrad.init();

    FreeWill::SoftmaxLogLossDerivative<FreeWill::CPU_NAIVE, float> softmaxLogLossDerivative;
    softmaxLogLossDerivative.setInputParameter("Output", &softmaxOutput);
    softmaxLogLossDerivative.setInputParameter("Label", &label);
    softmaxLogLossDerivative.setOutputParameter("InputGrad", &softmaxGrad);
    VERIFY_INIT(softmaxLogLossDerivative.init());

    FreeWill::DotProductWithBiasDerivative<FreeWill::CPU_NAIVE, float> dotProductWithBias2Derivative;
    dotProductWithBias2Derivative.setInputParameter("InputActivation", &fullyConnected1Output);
    dotProductWithBias2Derivative.setInputParameter("OutputDelta", &softmaxGrad);
    dotProductWithBias2Derivative.setInputParameter("Weight", &fullyConnected2Weight);

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> fullyConnected1OutputGrad({100,1});
    fullyConnected1OutputGrad.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> fullyConnected2WeightGrad({10,100});
    fullyConnected2WeightGrad.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> fullyConnected2BiasGrad({10});
    fullyConnected2BiasGrad.init();

    dotProductWithBias2Derivative.setOutputParameter("InputDelta", &fullyConnected1OutputGrad);
    dotProductWithBias2Derivative.setOutputParameter("BiasGrad", &fullyConnected2BiasGrad);
    dotProductWithBias2Derivative.setOutputParameter("WeightGrad", &fullyConnected2WeightGrad);

    VERIFY_INIT(dotProductWithBias2Derivative.init());

    FreeWill::ActivationDerivative<FreeWill::SIGMOID, FreeWill::CPU_NAIVE, float> sigmoidDerivative;
    sigmoidDerivative.setInputParameter("Output", &fullyConnected1Output);
    sigmoidDerivative.setInputParameter("OutputDelta", &fullyConnected1OutputGrad);
    sigmoidDerivative.setOutputParameter("InputDelta", &fullyConnected1OutputGrad);

    VERIFY_INIT(sigmoidDerivative.init());

    FreeWill::DotProductWithBiasDerivative<FreeWill::CPU_NAIVE, float> dotProductWithBias1Derivative;
    dotProductWithBias1Derivative.setInputParameter("InputActivation", &poolingOutput);
    dotProductWithBias1Derivative.setInputParameter("OutputDelta", &fullyConnected1OutputGrad);
    dotProductWithBias1Derivative.setInputParameter("Weight", &fullyConnected1Weight);

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> poolingOutputGrad({featureMapSize*12*12,1});
    poolingOutputGrad.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> fullyConnected1WeightGrad({100, featureMapSize*12*12});
    fullyConnected1WeightGrad.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> fullyConnected1BiasGrad({100});
    fullyConnected1BiasGrad.init();

    dotProductWithBias1Derivative.setOutputParameter("InputDelta", &poolingOutputGrad);
    dotProductWithBias1Derivative.setOutputParameter("BiasGrad", &fullyConnected1BiasGrad);
    dotProductWithBias1Derivative.setOutputParameter("WeightGrad", &fullyConnected1WeightGrad);

    VERIFY_INIT(dotProductWithBias1Derivative.init());


    poolingOutputGrad.reshape({featureMapSize,12,12,1});

    FreeWill::MaxPoolingDerivative<FreeWill::CPU_NAIVE, float> maxPoolingDerivative;
    maxPoolingDerivative.setInputParameter("OutputGrad", &poolingOutputGrad);
    maxPoolingDerivative.setInputParameter("SwitchX", &poolingSwitchX);
    maxPoolingDerivative.setInputParameter("SwitchY", &poolingSwitchY);

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> convOutputGrad({featureMapSize,24,24,1});
    convOutputGrad.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> convBiasGrad({featureMapSize});
    convBiasGrad.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> convFeatureMapGrad({1,5,5,featureMapSize});
    convFeatureMapGrad.init();

    maxPoolingDerivative.setOutputParameter("InputGrad", &convOutputGrad);
    VERIFY_INIT(maxPoolingDerivative.init());

    FreeWill::ActivationDerivative<FreeWill::SIGMOID, FreeWill::CPU_NAIVE, float> convSigmoidDerivative;
    convSigmoidDerivative.setInputParameter("Output", &convOutput);
    convSigmoidDerivative.setInputParameter("OutputDelta", &convOutputGrad);
    convSigmoidDerivative.setOutputParameter("InputDelta", &convOutputGrad);
    VERIFY_INIT(convSigmoidDerivative.init());

    FreeWill::ConvolutionDerivative<FreeWill::CPU_NAIVE, float> convDerivative;
    convDerivative.setInputParameter("PrevActivation", &image);
    convDerivative.setInputParameter("FeatureMap", &featureMap);
    convDerivative.setInputParameter("OutputGrad", &convOutputGrad);
   
    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> inputGrad({1,28,28,1});
    inputGrad.init();

    convDerivative.setOutputParameter("FeatureMapGrad", &convFeatureMapGrad);
    convDerivative.setOutputParameter("BiasGrad", &convBiasGrad);
    convDerivative.setOutputParameter("InputGrad", &inputGrad);

    VERIFY_INIT(convDerivative.init());


    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> batchConvWeight({1,5,5,featureMapSize});
    batchConvWeight.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> batchConvBias({featureMapSize});
    batchConvBias.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> batchFullyConnected1Weight({100,featureMapSize*12*12});
    batchFullyConnected1Weight.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> batchFullyConnected1Bias({100});
    batchFullyConnected1Bias.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> batchFullyConnected2Weight({10,100});
    batchFullyConnected2Weight.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> batchFullyConnected2Bias({10});
    batchFullyConnected2Bias.init();


    FreeWill::ElementwiseAdd<FreeWill::CPU_NAIVE, float> accumulateConvWeight;
    accumulateConvWeight.setInputParameter("Operand", &batchConvWeight);
    accumulateConvWeight.setInputParameter("Operand", &convFeatureMapGrad);
    accumulateConvWeight.setOutputParameter("Result", &batchConvWeight);
    VERIFY_INIT(accumulateConvWeight.init());

    FreeWill::ElementwiseAdd<FreeWill::CPU_NAIVE, float> accumulateConvBias;
    accumulateConvBias.setInputParameter("Operand", &batchConvBias);
    accumulateConvBias.setInputParameter("Operand", &convBiasGrad);
    accumulateConvBias.setOutputParameter("Result", &batchConvBias);
    VERIFY_INIT(accumulateConvBias.init());

    FreeWill::ElementwiseAdd<FreeWill::CPU_NAIVE, float> accumulateFullyConnected1Weight;
    accumulateFullyConnected1Weight.setInputParameter("Operand", &fullyConnected1WeightGrad);
    accumulateFullyConnected1Weight.setInputParameter("Operand", &batchFullyConnected1Weight);
    accumulateFullyConnected1Weight.setOutputParameter("Result", &batchFullyConnected1Weight);
    VERIFY_INIT(accumulateFullyConnected1Weight.init());

    FreeWill::ElementwiseAdd<FreeWill::CPU_NAIVE, float> accumulateFullyConnected2Weight;
    accumulateFullyConnected2Weight.setInputParameter("Operand", &batchFullyConnected2Weight);
    accumulateFullyConnected2Weight.setInputParameter("Operand", &fullyConnected2WeightGrad);
    accumulateFullyConnected2Weight.setOutputParameter("Result", &batchFullyConnected2Weight);
    VERIFY_INIT(accumulateFullyConnected2Weight.init());

    FreeWill::ElementwiseAdd<FreeWill::CPU_NAIVE, float> accumulateFullyConnected1Bias;
    accumulateFullyConnected1Bias.setInputParameter("Operand", &batchFullyConnected1Bias);
    accumulateFullyConnected1Bias.setInputParameter("Operand", &fullyConnected1BiasGrad);
    accumulateFullyConnected1Bias.setOutputParameter("Result", &batchFullyConnected1Bias);

    VERIFY_INIT(accumulateFullyConnected1Bias.init());

    FreeWill::ElementwiseAdd<FreeWill::CPU_NAIVE, float> accumulateFullyConnected2Bias;
    accumulateFullyConnected2Bias.setInputParameter("Operand", &batchFullyConnected2Bias);
    accumulateFullyConnected2Bias.setInputParameter("Operand", &fullyConnected2BiasGrad);
    accumulateFullyConnected2Bias.setOutputParameter("Result", &batchFullyConnected2Bias);

    VERIFY_INIT(accumulateFullyConnected2Bias.init());

    FreeWill::ElementwiseAdd<FreeWill::CPU_NAIVE, float> updateConvWeight;
    updateConvWeight.setInputParameter("Operand", &featureMap);
    updateConvWeight.setInputParameter("Operand", &batchConvWeight);
    updateConvWeight.setOutputParameter("Result", &featureMap);

    VERIFY_INIT(updateConvWeight.init());

    FreeWill::ElementwiseAdd<FreeWill::CPU_NAIVE, float> updateConvBias;

    updateConvBias.setInputParameter("Operand", &bias);
    updateConvBias.setInputParameter("Operand", &batchConvBias);
    updateConvBias.setOutputParameter("Result", &bias);

    VERIFY_INIT(updateConvBias.init());

    FreeWill::ElementwiseAdd<FreeWill::CPU_NAIVE, float> updateFullyConnected1Weight;
    updateFullyConnected1Weight.setInputParameter("Operand", &fullyConnected1Weight);
    updateFullyConnected1Weight.setInputParameter("Operand", &batchFullyConnected1Weight);
    updateFullyConnected1Weight.setOutputParameter("Result", &fullyConnected1Weight);

    VERIFY_INIT(updateFullyConnected1Weight.init());

    FreeWill::ElementwiseAdd<FreeWill::CPU_NAIVE, float> updateFullyConnected2Weight;
    updateFullyConnected2Weight.setInputParameter("Operand", &fullyConnected2Weight);
    updateFullyConnected2Weight.setInputParameter("Operand", &batchFullyConnected2Weight);
    updateFullyConnected2Weight.setOutputParameter("Result", &fullyConnected2Weight);

    VERIFY_INIT(updateFullyConnected2Weight.init());


    FreeWill::ElementwiseAdd<FreeWill::CPU_NAIVE, float> updateFullyConnected1Bias;
    updateFullyConnected1Bias.setInputParameter("Operand", &fullyConnected1Bias);
    updateFullyConnected1Bias.setInputParameter("Operand", &batchFullyConnected1Bias);
    updateFullyConnected1Bias.setOutputParameter("Result", &fullyConnected1Bias);

    VERIFY_INIT(updateFullyConnected1Bias.init());

    FreeWill::ElementwiseAdd<FreeWill::CPU_NAIVE, float> updateFullyConnected2Bias;
    updateFullyConnected2Bias.setInputParameter("Operand", &fullyConnected2Bias);
    updateFullyConnected2Bias.setInputParameter("Operand", &batchFullyConnected2Bias);
    updateFullyConnected2Bias.setOutputParameter("Result", &fullyConnected2Bias);

    VERIFY_INIT(updateFullyConnected2Bias.init());



    float learningRate = 0.1;

    //openData();
    //loadOneData(image, label);
    int batchSize = 10;
    float overallCost = 0.0;
    const int testInterval = 60000;

    float accuracy = 0.0;

    for(unsigned int e = 1;e<=60;++e)
    {
        openTrainData();

    
        for(unsigned int i = 1;i<=numOfImage;++i)
        {
            //openData();
            loadOneTrainData(image, label);

            //forward
            convolution.evaluate();
            convSigmoid.evaluate();
            //convReLU.evaluate();
            poolingOutput.reshape({featureMapSize,12,12,1});
            maxPooling.evaluate();
            poolingOutput.reshape({featureMapSize*12*12,1});
            fullyConnected1.evaluate();
            sigmoid1.evaluate();
            //ReLu1.evaluate();
            fullyConnected2.evaluate();
            softmaxLogLoss.evaluate();

            //qDebug() << "cost" << cost[0];
            overallCost += cost[0];
            //backward
            softmaxLogLossDerivative.evaluate();
            dotProductWithBias2Derivative.evaluate();
            sigmoidDerivative.evaluate();
            //reLUDerivative.evaluate();
            poolingOutputGrad.reshape({featureMapSize*12*12,1});
            dotProductWithBias1Derivative.evaluate();
            poolingOutputGrad.reshape({featureMapSize,12,12,1});
            maxPoolingDerivative.evaluate();
            convSigmoidDerivative.evaluate();
            //convReLUDerivative.evaluate();
            convDerivative.evaluate();


            accumulateConvWeight.evaluate();
            accumulateConvBias.evaluate();
            accumulateFullyConnected1Weight.evaluate();
            accumulateFullyConnected1Bias.evaluate();
            accumulateFullyConnected2Weight.evaluate();
            accumulateFullyConnected2Bias.evaluate();


            if (i%batchSize == 0)
            {
                qDebug() << e << i<< "cost" << overallCost / (float) batchSize << learningRate << accuracy;
                emit updateCost(overallCost / (float) batchSize);
                overallCost = 0.0;

                //update weight
                updateConvWeight.setRate(-learningRate/(float)batchSize);        
                updateConvWeight.evaluate();
                updateConvBias.setRate(-learningRate/(float)batchSize);
                updateConvBias.evaluate();
                updateFullyConnected1Weight.setRate(-learningRate/(float)batchSize);
                updateFullyConnected1Weight.evaluate();
                updateFullyConnected1Bias.setRate(-learningRate/(float)batchSize);
                updateFullyConnected1Bias.evaluate();
                updateFullyConnected2Weight.setRate(-learningRate/(float)batchSize);
                updateFullyConnected2Weight.evaluate();        
                updateFullyConnected2Bias.setRate(-learningRate/(float)batchSize);
                updateFullyConnected2Bias.evaluate();
            
                batchConvWeight.clear();
                batchConvBias.clear();
                batchFullyConnected1Weight.clear();
                batchFullyConnected2Weight.clear();
                batchFullyConnected1Bias.clear();
                batchFullyConnected2Bias.clear();

                if (i%60000 == 0)
                {
                    learningRate *= 0.95;
                }
            }

            //test
            //
            if (i % testInterval == 0)
            {
                unsigned int correct = 0;
                
                openTestData();

                for (unsigned int v = 0;v<numOfTestImage; ++v)
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
 
                    loadOneTestData(image, label);

                    //forward
                    convolution.evaluate();
                    convSigmoid.evaluate();
                    //convReLU.evaluate();
                    poolingOutput.reshape({featureMapSize,12,12,1});
                    maxPooling.evaluate();
                    poolingOutput.reshape({featureMapSize*12*12,1});
                    fullyConnected1.evaluate();
                    sigmoid1.evaluate();
                    //ReLu1.evaluate();
                    fullyConnected2.evaluate();


                    unsigned int maxIndex = 0;
                    float maxValue = fullyConnected2Output[0];
                    for(unsigned int e = 1; e < fullyConnected2Output.shape().size(); ++e)
                    {
                        if (maxValue < fullyConnected2Output[e])
                        {
                            maxValue = fullyConnected2Output[e];
                            maxIndex = e;
                        }
                    }

                    if ((float) maxIndex == label[0])
                    {
                        correct ++;
                    }

                 }

                 qDebug() << "Accuracy" << (accuracy = (float) correct / (float) numOfTestImage);

                 closeTestData();
             }

            //clean up
        
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

            //closeData();
       
           emit updateProgress(i / (float)(numOfImage), ((e-1)*numOfImage + i) / (60.0f*numOfImage)); 
        }
    
        /*if (e % 10000 == 0)
        {
            learningRate *= 0.8;
        }*/
        closeTrainData();
    }
    //closeData();
}

void MNIST::trainFullyConnectedModel()
{
    //openData();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> image({28*28,1});
    image.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, unsigned int> label({1});
    label.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> fullyConnected1Weight({100, 28*28});
    fullyConnected1Weight.init();
    fullyConnected1Weight.randomize();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> fullyConnected1Bias({100});
    fullyConnected1Bias.init();
    fullyConnected1Bias.randomize();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> fullyConnected1Output({100, 1});
    fullyConnected1Output.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> fullyConnected2Weight({10, 100});
    fullyConnected2Weight.init();
    fullyConnected2Weight.randomize();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> fullyConnected2Bias({10});
    fullyConnected2Bias.init();
    fullyConnected2Bias.randomize();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> fullyConnected2Output({10, 1});
    fullyConnected2Output.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> softmaxOutput({10,1});
    softmaxOutput.init();    

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> cost({1});
    cost.init();

    FreeWill::DotProductWithBias<FreeWill::CPU_NAIVE, float> fullyConnected1;
    fullyConnected1.setInputParameter("Input", &image);
    fullyConnected1.setInputParameter("Weight", &fullyConnected1Weight);
    fullyConnected1.setInputParameter("Bias", &fullyConnected1Bias);
    fullyConnected1.setOutputParameter("Output", &fullyConnected1Output);
    VERIFY_INIT(fullyConnected1.init());

    FreeWill::Activation<FreeWill::SIGMOID, FreeWill::CPU_NAIVE, float> sigmoid1;
    sigmoid1.setInputParameter("Input", &fullyConnected1Output);
    sigmoid1.setOutputParameter("Output", &fullyConnected1Output);
    VERIFY_INIT(sigmoid1.init());

    FreeWill::DotProductWithBias<FreeWill::CPU_NAIVE, float> fullyConnected2;
    fullyConnected2.setInputParameter("Input", &fullyConnected1Output);
    fullyConnected2.setInputParameter("Weight", &fullyConnected2Weight);
    fullyConnected2.setInputParameter("Bias", &fullyConnected2Bias);
    fullyConnected2.setOutputParameter("Output", &fullyConnected2Output);
    VERIFY_INIT(fullyConnected2.init());

    FreeWill::SoftmaxLogLoss<FreeWill::CPU_NAIVE, float> softmaxLogLoss;
    softmaxLogLoss.setInputParameter("Input", &fullyConnected2Output);
    softmaxLogLoss.setInputParameter("Label", &label);
    softmaxLogLoss.setOutputParameter("Output", &softmaxOutput);
    softmaxLogLoss.setOutputParameter("Cost", &cost);
    VERIFY_INIT(softmaxLogLoss.init());


    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> softmaxGrad({10,1});
    softmaxGrad.init();

    FreeWill::SoftmaxLogLossDerivative<FreeWill::CPU_NAIVE, float> softmaxLogLossDerivative;
    softmaxLogLossDerivative.setInputParameter("Output", &softmaxOutput);
    softmaxLogLossDerivative.setInputParameter("Label", &label);
    softmaxLogLossDerivative.setOutputParameter("InputGrad", &softmaxGrad);
    VERIFY_INIT(softmaxLogLossDerivative.init());

    FreeWill::DotProductWithBiasDerivative<FreeWill::CPU_NAIVE, float> dotProductWithBias2Derivative;
    dotProductWithBias2Derivative.setInputParameter("InputActivation", &fullyConnected1Output);
    dotProductWithBias2Derivative.setInputParameter("OutputDelta", &softmaxGrad);
    dotProductWithBias2Derivative.setInputParameter("Weight", &fullyConnected2Weight);

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> fullyConnected1OutputGrad({100,1});
    fullyConnected1OutputGrad.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> fullyConnected2WeightGrad({10,100});
    fullyConnected2WeightGrad.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> fullyConnected2BiasGrad({10});
    fullyConnected2BiasGrad.init();

    dotProductWithBias2Derivative.setOutputParameter("InputDelta", &fullyConnected1OutputGrad);
    dotProductWithBias2Derivative.setOutputParameter("BiasGrad", &fullyConnected2BiasGrad);
    dotProductWithBias2Derivative.setOutputParameter("WeightGrad", &fullyConnected2WeightGrad);

    VERIFY_INIT(dotProductWithBias2Derivative.init());

    FreeWill::ActivationDerivative<FreeWill::SIGMOID, FreeWill::CPU_NAIVE, float> sigmoidDerivative;
    sigmoidDerivative.setInputParameter("Output", &fullyConnected1Output);
    sigmoidDerivative.setInputParameter("OutputDelta", &fullyConnected1OutputGrad);
    sigmoidDerivative.setOutputParameter("InputDelta", &fullyConnected1OutputGrad);

    VERIFY_INIT(sigmoidDerivative.init());

    FreeWill::DotProductWithBiasDerivative<FreeWill::CPU_NAIVE, float> dotProductWithBias1Derivative;
    dotProductWithBias1Derivative.setInputParameter("InputActivation", &image);
    dotProductWithBias1Derivative.setInputParameter("OutputDelta", &fullyConnected1OutputGrad);
    dotProductWithBias1Derivative.setInputParameter("Weight", &fullyConnected1Weight);

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> fullyConnected1WeightGrad({100, 28*28});
    fullyConnected1WeightGrad.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> imageGrad({28*28, 1});
    imageGrad.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> fullyConnected1BiasGrad({100});
    fullyConnected1BiasGrad.init();

    dotProductWithBias1Derivative.setOutputParameter("InputDelta", &imageGrad);
    dotProductWithBias1Derivative.setOutputParameter("BiasGrad", &fullyConnected1BiasGrad);
    dotProductWithBias1Derivative.setOutputParameter("WeightGrad", &fullyConnected1WeightGrad);

    VERIFY_INIT(dotProductWithBias1Derivative.init());
   
    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> batchFullyConnected1Weight({100,28*28});
    batchFullyConnected1Weight.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> batchFullyConnected2Weight({10,100});
    batchFullyConnected2Weight.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> batchFullyConnected1Bias({100});
    batchFullyConnected1Bias.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> batchFullyConnected2Bias({10});
    batchFullyConnected2Bias.init();


    FreeWill::ElementwiseAdd<FreeWill::CPU_NAIVE, float> accumulateFullyConnected1Weight;
    accumulateFullyConnected1Weight.setInputParameter("Operand", &fullyConnected1WeightGrad);
    accumulateFullyConnected1Weight.setInputParameter("Operand", &batchFullyConnected1Weight);
    accumulateFullyConnected1Weight.setOutputParameter("Result", &batchFullyConnected1Weight);
    VERIFY_INIT(accumulateFullyConnected1Weight.init());

    FreeWill::ElementwiseAdd<FreeWill::CPU_NAIVE, float> accumulateFullyConnected1Bias;
    accumulateFullyConnected1Bias.setInputParameter("Operand", &fullyConnected1BiasGrad);
    accumulateFullyConnected1Bias.setInputParameter("Operand", &batchFullyConnected1Bias);
    accumulateFullyConnected1Bias.setOutputParameter("Result", &batchFullyConnected1Bias);
    VERIFY_INIT(accumulateFullyConnected1Bias.init());

    FreeWill::ElementwiseAdd<FreeWill::CPU_NAIVE, float> accumulateFullyConnected2Weight;
    accumulateFullyConnected2Weight.setInputParameter("Operand", &batchFullyConnected2Weight);
    accumulateFullyConnected2Weight.setInputParameter("Operand", &fullyConnected2WeightGrad);
    accumulateFullyConnected2Weight.setOutputParameter("Result", &batchFullyConnected2Weight);
    VERIFY_INIT(accumulateFullyConnected2Weight.init());

    FreeWill::ElementwiseAdd<FreeWill::CPU_NAIVE, float> accumulateFullyConnected2Bias;
    accumulateFullyConnected2Bias.setInputParameter("Operand", &batchFullyConnected2Bias);
    accumulateFullyConnected2Bias.setInputParameter("Operand", &fullyConnected2BiasGrad);
    accumulateFullyConnected2Bias.setOutputParameter("Result", &batchFullyConnected2Bias);
    VERIFY_INIT(accumulateFullyConnected2Bias.init());

    FreeWill::ElementwiseAdd<FreeWill::CPU_NAIVE, float> updateFullyConnected1Weight;
    updateFullyConnected1Weight.setInputParameter("Operand", &fullyConnected1Weight);
    updateFullyConnected1Weight.setInputParameter("Operand", &batchFullyConnected1Weight);
    updateFullyConnected1Weight.setOutputParameter("Result", &fullyConnected1Weight);

    VERIFY_INIT(updateFullyConnected1Weight.init());

    FreeWill::ElementwiseAdd<FreeWill::CPU_NAIVE, float> updateFullyConnected1Bias;
    updateFullyConnected1Bias.setInputParameter("Operand", &fullyConnected1Bias);
    updateFullyConnected1Bias.setInputParameter("Operand", &batchFullyConnected1Bias);
    updateFullyConnected1Bias.setOutputParameter("Result", &fullyConnected1Bias);

    VERIFY_INIT(updateFullyConnected1Bias.init());

    FreeWill::ElementwiseAdd<FreeWill::CPU_NAIVE, float> updateFullyConnected2Weight;
    updateFullyConnected2Weight.setInputParameter("Operand", &fullyConnected2Weight);
    updateFullyConnected2Weight.setInputParameter("Operand", &batchFullyConnected2Weight);
    updateFullyConnected2Weight.setOutputParameter("Result", &fullyConnected2Weight);

    VERIFY_INIT(updateFullyConnected2Weight.init());

    FreeWill::ElementwiseAdd<FreeWill::CPU_NAIVE, float> updateFullyConnected2Bias;
    updateFullyConnected2Bias.setInputParameter("Operand", &fullyConnected2Bias);
    updateFullyConnected2Bias.setInputParameter("Operand", &batchFullyConnected2Bias);
    updateFullyConnected2Bias.setOutputParameter("Result", &fullyConnected2Bias);

    VERIFY_INIT(updateFullyConnected2Bias.init());

    float learningRate = 0.01;

    //openData();
    //loadOneData(image, label);
    int batchSize = 20;
    float overallCost = 0.0;
    const int testInterval = 2000;



    for(unsigned int e = 1;e<=60;++e)
    {
        openTrainData();

    
        for(unsigned int i = 1;i<=numOfImage;++i)
        {
            //openData();
            loadOneTrainData(image, label);

            fullyConnected1.evaluate();
            sigmoid1.evaluate();
            //ReLu1.evaluate();
            fullyConnected2.evaluate();
            softmaxLogLoss.evaluate();

            //qDebug() << "cost" << cost[0];
            overallCost += cost[0];
            //backward
            softmaxLogLossDerivative.evaluate();
            dotProductWithBias2Derivative.evaluate();
            sigmoidDerivative.evaluate();
            dotProductWithBias1Derivative.evaluate();

            accumulateFullyConnected1Weight.evaluate();
            accumulateFullyConnected1Bias.evaluate();
            accumulateFullyConnected2Weight.evaluate();
            accumulateFullyConnected2Bias.evaluate();

            if (i%batchSize == 0)
            {
                qDebug() << e << i<< "cost" << overallCost / (float) batchSize << learningRate;
                emit updateCost(overallCost / (float) batchSize);
                overallCost = 0.0;

                //update weight
                updateFullyConnected1Weight.setRate(-learningRate/(float)batchSize);
                updateFullyConnected1Weight.evaluate();

                updateFullyConnected1Bias.setRate(-learningRate/(float)batchSize);
                updateFullyConnected1Bias.evaluate();
                
                updateFullyConnected2Weight.setRate(-learningRate/(float)batchSize);
                updateFullyConnected2Weight.evaluate();        

                updateFullyConnected2Bias.setRate(-learningRate/(float)batchSize);
                updateFullyConnected2Bias.evaluate();
            
                batchFullyConnected1Weight.clear();
                batchFullyConnected1Bias.clear();
                batchFullyConnected2Weight.clear();
                batchFullyConnected2Bias.clear();
           
               if (i%30000 == 0)
               {
                   learningRate *= 0.9;
               } 
            }

            //test
            //
            if (i % testInterval == 0)
            {
                unsigned int correct = 0;
                
                openTestData();

                for (unsigned int v = 0;v<numOfTestImage; ++v)
                {
                    fullyConnected1Output.clear();
                    fullyConnected2Output.clear();
                    softmaxOutput.clear();
                    cost.clear();
                    softmaxGrad.clear();
 
                    loadOneTestData(image, label);

                    //forward
                    fullyConnected1.evaluate();
                    sigmoid1.evaluate();
                    //ReLu1.evaluate();
                    fullyConnected2.evaluate();


                    unsigned int maxIndex = 0;
                    float maxValue = fullyConnected2Output[0];
                    for(unsigned int e = 1; e < fullyConnected2Output.shape().size(); ++e)
                    {
                        if (maxValue < fullyConnected2Output[e])
                        {
                            maxValue = fullyConnected2Output[e];
                            maxIndex = e;
                        }
                    }

                    if ((float) maxIndex == label[0])
                    {
                        correct ++;
                    }

                 }

                 qDebug() << "Accuracy" << (float) correct / (float) numOfTestImage;

                 closeTestData();
             }

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
            emit updateProgress(i / (float)(numOfImage), ((e-1)*numOfImage + i) / (60.0f*numOfImage)); 
        
        }
    
        /*if (e % 10000 == 0)
        {
            learningRate *= 0.8;
        }*/
        closeTrainData();
    }
    //closeData();
}

void MNIST::trainConvolutionalModelGPU()
{

    const unsigned int featureMapSize = 20;

    FreeWill::Tensor<FreeWill::GPU_CUDA, float> image({1,28,28,1});
    image.init();

    FreeWill::Tensor<FreeWill::GPU_CUDA, unsigned int> label({1});
    label.init();

    FreeWill::Tensor<FreeWill::GPU_CUDA, float> featureMap({1,5,5, featureMapSize});
    featureMap.init();
    featureMap.randomize();
    featureMap.copyFromHostToDevice();

    FreeWill::Tensor<FreeWill::GPU_CUDA, float> bias({featureMapSize});
    bias.init();
    bias.randomize();
    bias.copyFromHostToDevice();

    FreeWill::Tensor<FreeWill::GPU_CUDA, float> convOutput({featureMapSize,24,24,1});
    convOutput.init();

    FreeWill::Tensor<FreeWill::GPU_CUDA, float> poolingOutput({featureMapSize,12,12,1});
    poolingOutput.init();

    //FreeWill::Tensor<FreeWill::GPU_CUDA, unsigned int> poolingSwitchX({featureMapSize,12,12,1});
    //poolingSwitchX.init();

    //FreeWill::Tensor<FreeWill::GPU_CUDA, unsigned int> poolingSwitchY({featureMapSize,12,12,1});
    //poolingSwitchY.init();

    FreeWill::Tensor<FreeWill::GPU_CUDA, float> fullyConnected1Weight({100, featureMapSize*12*12});
    fullyConnected1Weight.init();
    fullyConnected1Weight.randomize();
    fullyConnected1Weight.copyFromHostToDevice();

    FreeWill::Tensor<FreeWill::GPU_CUDA, float> fullyConnected1Bias({100});
    fullyConnected1Bias.init();
    fullyConnected1Bias.randomize();
    fullyConnected1Bias.copyFromHostToDevice();

    FreeWill::Tensor<FreeWill::GPU_CUDA, float> fullyConnected1Output({100, 1});
    fullyConnected1Output.init();

    FreeWill::Tensor<FreeWill::GPU_CUDA, float> fullyConnected2Weight({10, 100});
    fullyConnected2Weight.init();
    fullyConnected2Weight.randomize();
    fullyConnected2Weight.copyFromHostToDevice();

    FreeWill::Tensor<FreeWill::GPU_CUDA, float> fullyConnected2Bias({10});
    fullyConnected2Bias.init();
    fullyConnected2Bias.randomize();
    fullyConnected2Bias.copyFromHostToDevice();

    FreeWill::Tensor<FreeWill::GPU_CUDA, float> fullyConnected2Output({10, 1});
    fullyConnected2Output.init();

    FreeWill::Tensor<FreeWill::GPU_CUDA, float> softmaxOutput({10,1});
    softmaxOutput.init();

    FreeWill::Tensor<FreeWill::GPU_CUDA, float> cost({1});
    cost.init();


    FreeWill::Convolution<FreeWill::GPU_CUDA, float> convolution;
    convolution.setInputParameter("Input", &image);
    convolution.setInputParameter("FeatureMap", &featureMap);
    convolution.setInputParameter("Bias", &bias);
    convolution.setOutputParameter("Output", &convOutput);
    VERIFY_INIT(convolution.init());

    FreeWill::Activation<FreeWill::SIGMOID, FreeWill::GPU_CUDA, float> convSigmoid;
    convSigmoid.setInputParameter("Input", &convOutput);
    convSigmoid.setOutputParameter("Output", &convOutput);
    VERIFY_INIT(convSigmoid.init());

    FreeWill::MaxPooling<FreeWill::GPU_CUDA, float> maxPooling;
    maxPooling.setInputParameter("Input", &convOutput);
    maxPooling.setOutputParameter("Output", &poolingOutput);
    //maxPooling.setOutputParameter("SwitchX", &poolingSwitchX);
    //maxPooling.setOutputParameter("SwitchY", &poolingSwitchY);
    VERIFY_INIT(maxPooling.init());


    poolingOutput.reshape({featureMapSize*12*12, 1});

    FreeWill::DotProductWithBias<FreeWill::GPU_CUDA, float> fullyConnected1;
    fullyConnected1.setInputParameter("Input", &poolingOutput);
    fullyConnected1.setInputParameter("Weight", &fullyConnected1Weight);
    fullyConnected1.setInputParameter("Bias", &fullyConnected1Bias);
    fullyConnected1.setOutputParameter("Output", &fullyConnected1Output);
    VERIFY_INIT(fullyConnected1.init());

    FreeWill::Activation<FreeWill::SIGMOID, FreeWill::GPU_CUDA, float> sigmoid1;
    sigmoid1.setInputParameter("Input", &fullyConnected1Output);
    sigmoid1.setOutputParameter("Output", &fullyConnected1Output);
    VERIFY_INIT(sigmoid1.init());


    FreeWill::DotProductWithBias<FreeWill::GPU_CUDA, float> fullyConnected2;
    fullyConnected2.setInputParameter("Input", &fullyConnected1Output);
    fullyConnected2.setInputParameter("Weight", &fullyConnected2Weight);
    fullyConnected2.setInputParameter("Bias", &fullyConnected2Bias);
    fullyConnected2.setOutputParameter("Output", &fullyConnected2Output);
    VERIFY_INIT(fullyConnected2.init());

    FreeWill::SoftmaxLogLoss<FreeWill::GPU_CUDA, float> softmaxLogLoss;
    softmaxLogLoss.setInputParameter("Input", &fullyConnected2Output);
    softmaxLogLoss.setInputParameter("Label", &label);
    softmaxLogLoss.setOutputParameter("Output", &softmaxOutput);
    softmaxLogLoss.setOutputParameter("Cost", &cost);
    VERIFY_INIT(softmaxLogLoss.init());


    FreeWill::Tensor<FreeWill::GPU_CUDA, float> softmaxGrad({10,1});
    softmaxGrad.init();

    FreeWill::SoftmaxLogLossDerivative<FreeWill::GPU_CUDA, float> softmaxLogLossDerivative;
    softmaxLogLossDerivative.setInputParameter("Output", &softmaxOutput);
    softmaxLogLossDerivative.setInputParameter("Label", &label);
    softmaxLogLossDerivative.setOutputParameter("InputGrad", &softmaxGrad);
    VERIFY_INIT(softmaxLogLossDerivative.init());

    FreeWill::DotProductWithBiasDerivative<FreeWill::GPU_CUDA, float> dotProductWithBias2Derivative;
    dotProductWithBias2Derivative.setInputParameter("InputActivation", &fullyConnected1Output);
    dotProductWithBias2Derivative.setInputParameter("OutputDelta", &softmaxGrad);
    dotProductWithBias2Derivative.setInputParameter("Weight", &fullyConnected2Weight);

    FreeWill::Tensor<FreeWill::GPU_CUDA, float> fullyConnected1OutputGrad({100,1});
    fullyConnected1OutputGrad.init();

    FreeWill::Tensor<FreeWill::GPU_CUDA, float> fullyConnected2WeightGrad({10,100});
    fullyConnected2WeightGrad.init();

    FreeWill::Tensor<FreeWill::GPU_CUDA, float> fullyConnected2BiasGrad({10});
    fullyConnected2BiasGrad.init();

    dotProductWithBias2Derivative.setOutputParameter("InputDelta", &fullyConnected1OutputGrad);
    dotProductWithBias2Derivative.setOutputParameter("BiasGrad", &fullyConnected2BiasGrad);
    dotProductWithBias2Derivative.setOutputParameter("WeightGrad", &fullyConnected2WeightGrad);

    VERIFY_INIT(dotProductWithBias2Derivative.init());

    FreeWill::ActivationDerivative<FreeWill::SIGMOID, FreeWill::GPU_CUDA, float> sigmoidDerivative;
    sigmoidDerivative.setInputParameter("Output", &fullyConnected1Output);
    sigmoidDerivative.setInputParameter("OutputDelta", &fullyConnected1OutputGrad);
    sigmoidDerivative.setOutputParameter("InputDelta", &fullyConnected1OutputGrad);

    VERIFY_INIT(sigmoidDerivative.init());

    FreeWill::DotProductWithBiasDerivative<FreeWill::GPU_CUDA, float> dotProductWithBias1Derivative;
    dotProductWithBias1Derivative.setInputParameter("InputActivation", &poolingOutput);
    dotProductWithBias1Derivative.setInputParameter("OutputDelta", &fullyConnected1OutputGrad);
    dotProductWithBias1Derivative.setInputParameter("Weight", &fullyConnected1Weight);

    FreeWill::Tensor<FreeWill::GPU_CUDA, float> poolingOutputGrad({featureMapSize*12*12,1});
    poolingOutputGrad.init();

    FreeWill::Tensor<FreeWill::GPU_CUDA, float> fullyConnected1WeightGrad({100, featureMapSize*12*12});
    fullyConnected1WeightGrad.init();

    FreeWill::Tensor<FreeWill::GPU_CUDA, float> fullyConnected1BiasGrad({100});
    fullyConnected1BiasGrad.init();

    dotProductWithBias1Derivative.setOutputParameter("InputDelta", &poolingOutputGrad);
    dotProductWithBias1Derivative.setOutputParameter("BiasGrad", &fullyConnected1BiasGrad);
    dotProductWithBias1Derivative.setOutputParameter("WeightGrad", &fullyConnected1WeightGrad);

    VERIFY_INIT(dotProductWithBias1Derivative.init());


    poolingOutputGrad.reshape({featureMapSize,12,12,1});

    FreeWill::MaxPoolingDerivative<FreeWill::GPU_CUDA, float> maxPoolingDerivative;
    maxPoolingDerivative.setInputParameter("OutputGrad", &poolingOutputGrad);
    //maxPoolingDerivative.setInputParameter("SwitchX", &poolingSwitchX);
    //maxPoolingDerivative.setInputParameter("SwitchY", &poolingSwitchY);
    maxPoolingDerivative.setInputParameter("Input", &convOutput);
    maxPoolingDerivative.setInputParameter("Output", &poolingOutput);

    FreeWill::Tensor<FreeWill::GPU_CUDA, float> convOutputGrad({featureMapSize,24,24,1});
    convOutputGrad.init();

    FreeWill::Tensor<FreeWill::GPU_CUDA, float> convBiasGrad({featureMapSize});
    convBiasGrad.init();

    FreeWill::Tensor<FreeWill::GPU_CUDA, float> convFeatureMapGrad({1,5,5,featureMapSize});
    convFeatureMapGrad.init();

    maxPoolingDerivative.setOutputParameter("InputGrad", &convOutputGrad);
    VERIFY_INIT(maxPoolingDerivative.init());

    FreeWill::ActivationDerivative<FreeWill::SIGMOID, FreeWill::GPU_CUDA, float> convSigmoidDerivative;
    convSigmoidDerivative.setInputParameter("Output", &convOutput);
    convSigmoidDerivative.setInputParameter("OutputDelta", &convOutputGrad);
    convSigmoidDerivative.setOutputParameter("InputDelta", &convOutputGrad);
    VERIFY_INIT(convSigmoidDerivative.init());

    FreeWill::ConvolutionDerivative<FreeWill::GPU_CUDA, float> convDerivative;
    convDerivative.setInputParameter("PrevActivation", &image);
    convDerivative.setInputParameter("FeatureMap", &featureMap);
    convDerivative.setInputParameter("OutputGrad", &convOutputGrad);

    FreeWill::Tensor<FreeWill::GPU_CUDA, float> inputGrad({1,28,28,1});
    inputGrad.init();

    convDerivative.setOutputParameter("FeatureMapGrad", &convFeatureMapGrad);
    convDerivative.setOutputParameter("BiasGrad", &convBiasGrad);
    convDerivative.setOutputParameter("InputGrad", &inputGrad);

    VERIFY_INIT(convDerivative.init());


    FreeWill::Tensor<FreeWill::GPU_CUDA, float> batchConvWeight({1,5,5,featureMapSize});
    batchConvWeight.init();

    FreeWill::Tensor<FreeWill::GPU_CUDA, float> batchConvBias({featureMapSize});
    batchConvBias.init();

    FreeWill::Tensor<FreeWill::GPU_CUDA, float> batchFullyConnected1Weight({100,featureMapSize*12*12});
    batchFullyConnected1Weight.init();

    FreeWill::Tensor<FreeWill::GPU_CUDA, float> batchFullyConnected1Bias({100});
    batchFullyConnected1Bias.init();

    FreeWill::Tensor<FreeWill::GPU_CUDA, float> batchFullyConnected2Weight({10,100});
    batchFullyConnected2Weight.init();

    FreeWill::Tensor<FreeWill::GPU_CUDA, float> batchFullyConnected2Bias({10});
    batchFullyConnected2Bias.init();


    FreeWill::ElementwiseAdd<FreeWill::GPU_CUDA, float> accumulateConvWeight;
    accumulateConvWeight.setInputParameter("Operand", &batchConvWeight);
    accumulateConvWeight.setInputParameter("Operand", &convFeatureMapGrad);
    accumulateConvWeight.setOutputParameter("Result", &batchConvWeight);
    VERIFY_INIT(accumulateConvWeight.init());

    FreeWill::ElementwiseAdd<FreeWill::GPU_CUDA, float> accumulateConvBias;
    accumulateConvBias.setInputParameter("Operand", &batchConvBias);
    accumulateConvBias.setInputParameter("Operand", &convBiasGrad);
    accumulateConvBias.setOutputParameter("Result", &batchConvBias);
    VERIFY_INIT(accumulateConvBias.init());

    FreeWill::ElementwiseAdd<FreeWill::GPU_CUDA, float> accumulateFullyConnected1Weight;
    accumulateFullyConnected1Weight.setInputParameter("Operand", &fullyConnected1WeightGrad);
    accumulateFullyConnected1Weight.setInputParameter("Operand", &batchFullyConnected1Weight);
    accumulateFullyConnected1Weight.setOutputParameter("Result", &batchFullyConnected1Weight);
    VERIFY_INIT(accumulateFullyConnected1Weight.init());

    FreeWill::ElementwiseAdd<FreeWill::GPU_CUDA, float> accumulateFullyConnected2Weight;
    accumulateFullyConnected2Weight.setInputParameter("Operand", &batchFullyConnected2Weight);
    accumulateFullyConnected2Weight.setInputParameter("Operand", &fullyConnected2WeightGrad);
    accumulateFullyConnected2Weight.setOutputParameter("Result", &batchFullyConnected2Weight);
    VERIFY_INIT(accumulateFullyConnected2Weight.init());

    FreeWill::ElementwiseAdd<FreeWill::GPU_CUDA, float> accumulateFullyConnected1Bias;
    accumulateFullyConnected1Bias.setInputParameter("Operand", &batchFullyConnected1Bias);
    accumulateFullyConnected1Bias.setInputParameter("Operand", &fullyConnected1BiasGrad);
    accumulateFullyConnected1Bias.setOutputParameter("Result", &batchFullyConnected1Bias);

    VERIFY_INIT(accumulateFullyConnected1Bias.init());

    FreeWill::ElementwiseAdd<FreeWill::GPU_CUDA, float> accumulateFullyConnected2Bias;
    accumulateFullyConnected2Bias.setInputParameter("Operand", &batchFullyConnected2Bias);
    accumulateFullyConnected2Bias.setInputParameter("Operand", &fullyConnected2BiasGrad);
    accumulateFullyConnected2Bias.setOutputParameter("Result", &batchFullyConnected2Bias);

    VERIFY_INIT(accumulateFullyConnected2Bias.init());

    FreeWill::ElementwiseAdd<FreeWill::GPU_CUDA, float> updateConvWeight;
    updateConvWeight.setInputParameter("Operand", &featureMap);
    updateConvWeight.setInputParameter("Operand", &batchConvWeight);
    updateConvWeight.setOutputParameter("Result", &featureMap);

    VERIFY_INIT(updateConvWeight.init());

    FreeWill::ElementwiseAdd<FreeWill::GPU_CUDA, float> updateConvBias;

    updateConvBias.setInputParameter("Operand", &bias);
    updateConvBias.setInputParameter("Operand", &batchConvBias);
    updateConvBias.setOutputParameter("Result", &bias);

    VERIFY_INIT(updateConvBias.init());

    FreeWill::ElementwiseAdd<FreeWill::GPU_CUDA, float> updateFullyConnected1Weight;
    updateFullyConnected1Weight.setInputParameter("Operand", &fullyConnected1Weight);
    updateFullyConnected1Weight.setInputParameter("Operand", &batchFullyConnected1Weight);
    updateFullyConnected1Weight.setOutputParameter("Result", &fullyConnected1Weight);

    VERIFY_INIT(updateFullyConnected1Weight.init());

    FreeWill::ElementwiseAdd<FreeWill::GPU_CUDA, float> updateFullyConnected2Weight;
    updateFullyConnected2Weight.setInputParameter("Operand", &fullyConnected2Weight);
    updateFullyConnected2Weight.setInputParameter("Operand", &batchFullyConnected2Weight);
    updateFullyConnected2Weight.setOutputParameter("Result", &fullyConnected2Weight);

    VERIFY_INIT(updateFullyConnected2Weight.init());


    FreeWill::ElementwiseAdd<FreeWill::GPU_CUDA, float> updateFullyConnected1Bias;
    updateFullyConnected1Bias.setInputParameter("Operand", &fullyConnected1Bias);
    updateFullyConnected1Bias.setInputParameter("Operand", &batchFullyConnected1Bias);
    updateFullyConnected1Bias.setOutputParameter("Result", &fullyConnected1Bias);

    VERIFY_INIT(updateFullyConnected1Bias.init());

    FreeWill::ElementwiseAdd<FreeWill::GPU_CUDA, float> updateFullyConnected2Bias;
    updateFullyConnected2Bias.setInputParameter("Operand", &fullyConnected2Bias);
    updateFullyConnected2Bias.setInputParameter("Operand", &batchFullyConnected2Bias);
    updateFullyConnected2Bias.setOutputParameter("Result", &fullyConnected2Bias);

    VERIFY_INIT(updateFullyConnected2Bias.init());



    float learningRate = 0.1;

    //openData();
    //loadOneData(image, label);
    int batchSize = 10;
    float overallCost = 0.0;
    const int testInterval = 60000;

    float accuracy = 0.0;

    for(unsigned int e = 1;e<=60;++e)
    {
        openTrainData();


        for(unsigned int i = 1;i<=numOfImage;++i)
        {
            //openData();
            loadOneTrainDataGPU(image, label);

            //forward
            convolution.evaluate();
            convSigmoid.evaluate();
            //convReLU.evaluate();
            poolingOutput.reshape({featureMapSize,12,12,1});
            maxPooling.evaluate();
            poolingOutput.reshape({featureMapSize*12*12,1});
            fullyConnected1.evaluate();
            sigmoid1.evaluate();
            //ReLu1.evaluate();
            fullyConnected2.evaluate();
            softmaxLogLoss.evaluate();

            //qDebug() << "cost" << cost[0];
            cost.copyFromDeviceToHost();
            overallCost += cost[0];
            //backward
            softmaxLogLossDerivative.evaluate();
            dotProductWithBias2Derivative.evaluate();
            sigmoidDerivative.evaluate();
            //reLUDerivative.evaluate();
            poolingOutputGrad.reshape({featureMapSize*12*12,1});
            dotProductWithBias1Derivative.evaluate();
            poolingOutputGrad.reshape({featureMapSize,12,12,1});
            maxPoolingDerivative.evaluate();
            convSigmoidDerivative.evaluate();
            //convReLUDerivative.evaluate();
            convDerivative.evaluate();


            accumulateConvWeight.evaluate();
            accumulateConvBias.evaluate();
            accumulateFullyConnected1Weight.evaluate();
            accumulateFullyConnected1Bias.evaluate();
            accumulateFullyConnected2Weight.evaluate();
            accumulateFullyConnected2Bias.evaluate();


            if (i%batchSize == 0)
            {
                qDebug() << e << i<< "cost" << overallCost / (float) batchSize << learningRate << accuracy;
                emit updateCost(overallCost / (float) batchSize);
                overallCost = 0.0;

                //update weight
                updateConvWeight.setRate(-learningRate/(float)batchSize);
                updateConvWeight.evaluate();
                updateConvBias.setRate(-learningRate/(float)batchSize);
                updateConvBias.evaluate();
                updateFullyConnected1Weight.setRate(-learningRate/(float)batchSize);
                updateFullyConnected1Weight.evaluate();
                updateFullyConnected1Bias.setRate(-learningRate/(float)batchSize);
                updateFullyConnected1Bias.evaluate();
                updateFullyConnected2Weight.setRate(-learningRate/(float)batchSize);
                updateFullyConnected2Weight.evaluate();
                updateFullyConnected2Bias.setRate(-learningRate/(float)batchSize);
                updateFullyConnected2Bias.evaluate();

                batchConvWeight.clear();
                batchConvBias.clear();
                batchFullyConnected1Weight.clear();
                batchFullyConnected2Weight.clear();
                batchFullyConnected1Bias.clear();
                batchFullyConnected2Bias.clear();

                if (i%60000 == 0)
                {
                    learningRate *= 0.95;
                }
            }

            //test
            //
            if (i % testInterval == 0)
            {
                unsigned int correct = 0;

                openTestData();

                for (unsigned int v = 0;v<numOfTestImage; ++v)
                {
                    convOutput.clear();
                    poolingOutput.clear();
                    //poolingSwitchX.clear();
                    //poolingSwitchY.clear();
                    fullyConnected1Output.clear();
                    fullyConnected2Output.clear();
                    softmaxOutput.clear();
                    cost.clear();
                    softmaxGrad.clear();

                    loadOneTestDataGPU(image, label);


                    //forward
                    convolution.evaluate();
                    convSigmoid.evaluate();
                    //convReLU.evaluate();
                    poolingOutput.reshape({featureMapSize,12,12,1});
                    maxPooling.evaluate();
                    poolingOutput.reshape({featureMapSize*12*12,1});
                    fullyConnected1.evaluate();
                    sigmoid1.evaluate();
                    //ReLu1.evaluate();
                    fullyConnected2.evaluate();


                    fullyConnected2Output.copyFromDeviceToHost();
                    unsigned int maxIndex = 0;
                    float maxValue = fullyConnected2Output[0];

                    for(unsigned int e = 1; e < fullyConnected2Output.shape().size(); ++e)
                    {
                        if (maxValue < fullyConnected2Output[e])
                        {
                            maxValue = fullyConnected2Output[e];
                            maxIndex = e;
                        }
                    }

                    if ((float) maxIndex == label[0])
                    {
                        correct ++;
                    }

                 }

                 qDebug() << "Accuracy" << (accuracy = (float) correct / (float) numOfTestImage);

                 closeTestData();
             }

            //clean up

            convOutput.clear();
            poolingOutput.clear();
            //poolingSwitchX.clear();
            //poolingSwitchY.clear();
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

            //closeData();

           emit updateProgress(i / (float)(numOfImage), ((e-1)*numOfImage + i) / (60.0f*numOfImage));
        }

        /*if (e % 10000 == 0)
        {
            learningRate *= 0.8;
        }*/
        closeTrainData();
    }
    //closeData();
}


void MNIST::run()
{

    srand(/*time(NULL)*/0);

    FreeWill::Context<FreeWill::GPU>::getSingleton().open();

    trainConvolutionalModelGPU();
    FreeWill::Context<FreeWill::GPU>::getSingleton().close();
return;
    if (false)
    {
        trainConvolutionalModel();
    }
    else
    {
        trainFullyConnectedModel();
    }
}
