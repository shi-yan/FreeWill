#include "endian.h"
#include "Tensor/Tensor.h"
#include "Operator/MaxPooling.h"
#include "Operator/MaxPoolingDerivative.h"
#include <cstdio>
#include "Operator/ReLU.h"
#include "Operator/Convolution.h"
#include "Operator/ConvolutionDerivative.h"
#include "Operator/ReLUDerivative.h"
#include "Operator/Softmax.h"
#include "Operator/SoftmaxDerivative.h"
#include "Operator/DotProductWithBias.h"
#include "Operator/DotProductWithBiasDerivative.h"
#include "Operator/Sigmoid.h"
#include "Operator/SigmoidDerivative.h"
#include "Operator/ElementwiseProduct.h"
#include <QDebug>
#include "Operator/ElementwiseAdd.h"

static FILE *datafp = 0;
static FILE *labelfp = 0;
static unsigned int numOfImage = 0;
static unsigned int numOfRow = 0;
static unsigned int numOfColumn = 0;
static unsigned int labelCount = 0;


void openData()
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

//    printf("magic number: %x\n", magicNum);
//    printf("num of image: %d\n", numOfImage);
//    printf("num of row: %d\n", numOfRow);
//    printf("num of column: %d\n", numOfColumn);
//    printf("magic number label: %x\n", magicNumLabel);
//    printf("num of label: %d\n", labelCount);   
}

void closeData()
{
    fclose(datafp);
    fclose(labelfp);
}


void loadOneData(FreeWill::Tensor<FreeWill::CPU, float> &image, FreeWill::Tensor<FreeWill::CPU, unsigned int> &label)
{
    
    for(unsigned int y = 0 ; y < numOfRow; ++y)
    {
        for(unsigned int x = 0;x< numOfColumn; ++x)
        {
            unsigned char pixel = 0;
            fread(&pixel, sizeof(unsigned char), 1, datafp);
            image[numOfColumn * y + x] = (float) pixel / 255.0f;
            //printf("%3d,", pixel);
        }
        //printf("\n");
    }
    unsigned char _label = 0;
    fread(&_label, sizeof(unsigned char), 1, labelfp);
    //printf("lable: %d\n", label);
    label[0] = _label;
}

int main()
{
    //openData();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> image({1,28,28,1});
    image.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, unsigned int> label({1});
    label.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> featureMap({1,5,5,20});
    featureMap.init();
    featureMap.randomize();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> bias({20});
    bias.init();
    bias.randomize();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> convOutput({20,24,24,1});
    convOutput.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> poolingOutput({20,12,12,1});
    poolingOutput.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, unsigned int> poolingSwitchX({20,12,12,1});
    poolingSwitchX.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, unsigned int> poolingSwitchY({20,12,12,1});
    poolingSwitchY.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> fullyConnected1Weight({100, 20*12*12+1});
    fullyConnected1Weight.init();
    fullyConnected1Weight.randomize();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> fullyConnected1Output({100, 1});
    fullyConnected1Output.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> fullyConnected2Weight({10, 100+1});
    fullyConnected2Weight.init();
    fullyConnected2Weight.randomize();

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

    FreeWill::Sigmoid<FreeWill::CPU_NAIVE, float> convSigmoid;
    convSigmoid.setInputParameter("Input", &convOutput);
    convSigmoid.setOutputParameter("Output", &convOutput);
    VERIFY_INIT(convSigmoid.init());
    
    FreeWill::MaxPooling<FreeWill::CPU_NAIVE, float> maxPooling;
    maxPooling.setInputParameter("Input", &convOutput);
    maxPooling.setOutputParameter("Output", &poolingOutput);
    maxPooling.setOutputParameter("SwitchX", &poolingSwitchX);
    maxPooling.setOutputParameter("SwitchY", &poolingSwitchY);
    VERIFY_INIT(maxPooling.init());


    poolingOutput.reshape({20*12*12, 1});

    FreeWill::DotProductWithBias<FreeWill::CPU_NAIVE, float> fullyConnected1;
    fullyConnected1.setInputParameter("Input", &poolingOutput);
    fullyConnected1.setInputParameter("Weight", &fullyConnected1Weight);
    fullyConnected1.setOutputParameter("Output", &fullyConnected1Output);
    VERIFY_INIT(fullyConnected1.init());

    FreeWill::Sigmoid<FreeWill::CPU_NAIVE, float> sigmoid1;
    sigmoid1.setInputParameter("Input", &fullyConnected1Output);
    sigmoid1.setOutputParameter("Output", &fullyConnected1Output);
    VERIFY_INIT(sigmoid1.init());

    FreeWill::DotProductWithBias<FreeWill::CPU_NAIVE, float> fullyConnected2;
    fullyConnected2.setInputParameter("Input", &fullyConnected1Output);
    fullyConnected2.setInputParameter("Weight", &fullyConnected2Weight);
    fullyConnected2.setOutputParameter("Output", &fullyConnected2Output);
    VERIFY_INIT(fullyConnected2.init());

    FreeWill::Softmax<FreeWill::CPU_NAIVE, float> softmax;
    softmax.setInputParameter("Input", &fullyConnected2Output);
    softmax.setInputParameter("Label", &label);
    softmax.setOutputParameter("Output", &softmaxOutput);
    softmax.setOutputParameter("Cost", &cost);
    VERIFY_INIT(softmax.init());


    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> softmaxGrad({10,1});
    softmaxGrad.init();

    FreeWill::SoftmaxDerivative<FreeWill::CPU_NAIVE, float> softmaxDerivative;
    softmaxDerivative.setInputParameter("Output", &softmaxOutput);
    softmaxDerivative.setInputParameter("Label", &label);
    softmaxDerivative.setOutputParameter("InputGrad", &softmaxGrad);
    VERIFY_INIT(softmaxDerivative.init());

    FreeWill::DotProductWithBiasDerivative<FreeWill::CPU_NAIVE, float> dotProductWithBias2Derivative;
    dotProductWithBias2Derivative.setInputParameter("PrevActivation", &fullyConnected1Output);
    dotProductWithBias2Derivative.setInputParameter("OutputGrad", &softmaxGrad);
    dotProductWithBias2Derivative.setInputParameter("Weight", &fullyConnected2Weight);

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> fullyConnected1OutputGrad({100,1});
    fullyConnected1OutputGrad.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> fullyConnected2WeightGrad({10,100+1,1});
    fullyConnected2WeightGrad.init();

    dotProductWithBias2Derivative.setOutputParameter("InputGrad", &fullyConnected1OutputGrad);
    dotProductWithBias2Derivative.setOutputParameter("WeightGrad", &fullyConnected2WeightGrad);

    VERIFY_INIT(dotProductWithBias2Derivative.init());

    FreeWill::SigmoidDerivative<FreeWill::CPU_NAIVE, float> sigmoidDerivative;
    sigmoidDerivative.setInputParameter("Input", &fullyConnected1Output);
    sigmoidDerivative.setOutputParameter("Output", &fullyConnected1Output);

    VERIFY_INIT(sigmoidDerivative.init());

    FreeWill::ElementwiseProduct<FreeWill::CPU_NAIVE, float> fullyConnected1OutputGradTimesSigGrad;
    fullyConnected1OutputGradTimesSigGrad.setInputParameter("OperandA", &fullyConnected1Output);
    fullyConnected1OutputGradTimesSigGrad.setInputParameter("OperandB", &fullyConnected1OutputGrad);
    fullyConnected1OutputGradTimesSigGrad.setOutputParameter("Output", &fullyConnected1OutputGrad);

    VERIFY_INIT(fullyConnected1OutputGradTimesSigGrad.init());

    FreeWill::DotProductWithBiasDerivative<FreeWill::CPU_NAIVE, float> dotProductWithBias1Derivative;
    dotProductWithBias1Derivative.setInputParameter("PrevActivation", &poolingOutput);
    dotProductWithBias1Derivative.setInputParameter("OutputGrad", &fullyConnected1OutputGrad);
    dotProductWithBias1Derivative.setInputParameter("Weight", &fullyConnected1Weight);

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> poolingOutputGrad({20*12*12,1});
    poolingOutputGrad.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> fullyConnected1WeightGrad({100, 20*12*12+1,1});
    fullyConnected1WeightGrad.init();

    dotProductWithBias1Derivative.setOutputParameter("InputGrad", &poolingOutputGrad);
    dotProductWithBias1Derivative.setOutputParameter("WeightGrad", &fullyConnected1WeightGrad);

    VERIFY_INIT(dotProductWithBias1Derivative.init());


    poolingOutputGrad.reshape({20,12,12,1});

    FreeWill::MaxPoolingDerivative<FreeWill::CPU_NAIVE, float> maxPoolingDerivative;
    maxPoolingDerivative.setInputParameter("OutputGrad", &poolingOutputGrad);
    maxPoolingDerivative.setInputParameter("SwitchX", &poolingSwitchX);
    maxPoolingDerivative.setInputParameter("SwitchY", &poolingSwitchY);

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> convOutputGrad({20,24,24,1});
    convOutputGrad.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> convBiasGrad({20});
    convBiasGrad.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> convFeatureMapGrad({1,5,5,20});
    convFeatureMapGrad.init();

    maxPoolingDerivative.setOutputParameter("InputGrad", &convOutputGrad);
    VERIFY_INIT(maxPoolingDerivative.init());

    FreeWill::SigmoidDerivative<FreeWill::CPU_NAIVE, float> convSigmoidDerivative;
    convSigmoidDerivative.setInputParameter("Input", &convOutput);
    convSigmoidDerivative.setOutputParameter("Output", &convOutput);
    VERIFY_INIT(convSigmoidDerivative.init());

    FreeWill::ElementwiseProduct<FreeWill::CPU_NAIVE, float> convSigmoidDerivativeTimesOutputGrad;
    convSigmoidDerivativeTimesOutputGrad.setInputParameter("OperandA", &convOutput);
    convSigmoidDerivativeTimesOutputGrad.setInputParameter("OperandB", &convOutputGrad);
    convSigmoidDerivativeTimesOutputGrad.setOutputParameter("Output", &convOutputGrad);

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

    FreeWill::ElementwiseAdd<FreeWill::CPU_NAIVE, float> updateConvWeight;
    updateConvWeight.setInputParameter("Operand", &featureMap);
    updateConvWeight.setInputParameter("Operand", &convFeatureMapGrad);
    updateConvWeight.setOutputParameter("Result", &featureMap);

    VERIFY_INIT(updateConvWeight.init());

    FreeWill::ElementwiseAdd<FreeWill::CPU_NAIVE, float> updateConvBias;

    updateConvBias.setInputParameter("Operand", &bias);
    updateConvBias.setInputParameter("Operand", &convBiasGrad);
    updateConvBias.setOutputParameter("Result", &bias);

    VERIFY_INIT(updateConvBias.init());

    FreeWill::ElementwiseAdd<FreeWill::CPU_NAIVE, float> updateFullyConnected1Weight;
    updateFullyConnected1Weight.setInputParameter("Operand", &fullyConnected1Weight);
    updateFullyConnected1Weight.setInputParameter("Operand", &fullyConnected1WeightGrad);
    updateFullyConnected1Weight.setOutputParameter("Result", &fullyConnected1Weight);

    VERIFY_INIT(updateFullyConnected1Weight.init());

    FreeWill::ElementwiseAdd<FreeWill::CPU_NAIVE, float> updateFullyConnected2Weight;
    updateFullyConnected2Weight.setInputParameter("Operand", &fullyConnected2Weight);
    updateFullyConnected2Weight.setInputParameter("Operand", &fullyConnected2WeightGrad);
    updateFullyConnected2Weight.setOutputParameter("Result", &fullyConnected2Weight);

    VERIFY_INIT(updateFullyConnected2Weight.init());

    float learningRate = 0.01;

    for(unsigned int e = 1;e<100000;++e)
    {
        openData();

    
        for(unsigned int i = 0;i<numOfImage;++i)
        {
            //openData();
            loadOneData(image, label);

            //forward
            convolution.evaluate();
            convSigmoid.evaluate();
            poolingOutput.reshape({20,12,12,1});
            maxPooling.evaluate();
            poolingOutput.reshape({20*12*12,1});
            fullyConnected1.evaluate();
            sigmoid1.evaluate();
            fullyConnected2.evaluate();
            softmax.evaluate();

            qDebug() << "cost" << cost[0];

            //backward
            softmaxDerivative.evaluate();
            dotProductWithBias2Derivative.evaluate();
            sigmoidDerivative.evaluate();
            fullyConnected1OutputGradTimesSigGrad.evaluate();
            poolingOutputGrad.reshape({20*12*12,1});
            dotProductWithBias1Derivative.evaluate();
            poolingOutputGrad.reshape({20,12,12,1});
            maxPooling.evaluate();
            convSigmoidDerivative.evaluate();
            convSigmoidDerivativeTimesOutputGrad.evaluate();
            convDerivative.evaluate();

            //update weight
            updateConvWeight.setRate(-learningRate);        
            updateConvWeight.evaluate();
            updateConvBias.setRate(-learningRate);
            updateConvBias.evaluate();
            updateFullyConnected1Weight.setRate(-learningRate);
            updateFullyConnected1Weight.evaluate();
            updateFullyConnected2Weight.setRate(-learningRate);
            updateFullyConnected2Weight.evaluate();        

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
            poolingOutputGrad.clear();
            convOutputGrad.clear();
            convBiasGrad.clear();
            convFeatureMapGrad.clear();
            inputGrad.clear();

            //closeData();
        
        }
    
        if (e % 10000 == 0)
        {
            learningRate *= 0.8;
        }
        closeData();
    }
    return 0;
}
