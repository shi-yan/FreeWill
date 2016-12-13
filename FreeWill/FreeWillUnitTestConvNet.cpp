#include "FreeWillUnitTest.h"
#include "Operator/Convolution.h"
#include "Operator/ConvolutionDerivative.h"
#include "Operator/ReLU.h"
#include "Operator/ReLUDerivative.h"
#include "Operator/Sigmoid.h"
#include "Operator/CrossEntropy.h"
#include "Operator/SigmoidCrossEntropyDerivative.h"
#include "Operator/SigmoidDerivative.h"


void FreeWillUnitTest::convNetTest()
{
    FreeWill::Tensor<FreeWill::CPU, float> input({3,5,5,1});
    input.init();
    
    
    FreeWill::Tensor<FreeWill::CPU, float> featureMaps({3,3,3,2});
    featureMaps.init();
    

    FreeWill::Tensor<FreeWill::CPU, float> output({2,3,3,1});
    output.init();

    FreeWill::Tensor<FreeWill::CPU, float> bias({2});
    bias.init();


    FreeWill::Convolution<FreeWill::CPU, float> convolution(2,2,1,1);
    convolution.setInputParameter("Input", &input);
    convolution.setInputParameter("FeatureMap", &featureMaps);
    convolution.setInputParameter("Bias", &bias);
    convolution.setOutputParameter("Output", &output);

    QVERIFY(convolution.init());

    float inputArray[] = {
            0,1,1, 0,1,1, 2,1,0, 0,1,2, 2,0,2,
            0,1,0,1,0,0,2,0,0,0,0,0,0,1,1,
            0,0,0,0,0,1,2,1,1,2,0,1,1,0,0,
            1,1,1,0,2,0,0,0,2,2,2,0,0,2,1,
            0,1,0,0,0,2,0,1,0,2,2,1,2,0,1
    };

    float featureMapArray[] = {
            0,0,1,-1,1,0,1,1,1,
            0,0,-1,1,-1,-1,0,1,-1,
            0,1,-1,-1,1,0,-1,0,-1,
            
            1,0,-1, 1,0,1, 1,-1,0,
            1,0,-1, 0,0,1, -1,0,1,
            -1,1,-1, -1,0,0, 1,1,-1
    };

    float biasArray[] = {1,0};

    unsigned int inputSize = input.shape().size();
    for(unsigned int i = 0; i< inputSize;++i)
    {
        input[i] = inputArray[i];
    }

    unsigned int featureMapsSize = featureMaps.shape().size();
    for(unsigned int i = 0; i< featureMapsSize;++i)
    {
        featureMaps[i] = featureMapArray[i];
    }

    unsigned int biasSize = bias.shape().size();
    for(unsigned int i = 0; i<biasSize;++i)
    {
        bias[i] = biasArray[i];
    }

    convolution.evaluate();

    float outputArray[] = {
       -1,3,-2,-2,0,0,
        2,3,-3,8,6,2,
        0,2,3,-1,3,5
    };

    unsigned int outputSize = output.shape().size();

    for(unsigned int i =0;i<outputSize;++i)
    {
        //qDebug() <<"output -"<< outputArray[i]<<";" << output[i];
        QVERIFY(outputArray[i] == output[i]);
    }

}

void FreeWillUnitTest::convDerivativeTest()
{
    FreeWill::Tensor<FreeWill::CPU, float> prevActivaion({3,5,5,1});
    prevActivaion.init();
    
    
    FreeWill::Tensor<FreeWill::CPU, float> featureMaps({3,3,3,2});
    featureMaps.init();
    

    FreeWill::Tensor<FreeWill::CPU, float> outputGrad({2,3,3,1});
    outputGrad.init();

    FreeWill::Tensor<FreeWill::CPU, float> inputGrad({3,5,5,1});
    inputGrad.init();

    FreeWill::Tensor<FreeWill::CPU, float> featureMapGrad({3,3,3,2});
    featureMapGrad.init();

    FreeWill::Tensor<FreeWill::CPU, float> fakeFeatureMapGrad({3,3,3,2});
    fakeFeatureMapGrad.init();

    FreeWill::Tensor<FreeWill::CPU, float> biasGrad({2});
    biasGrad.init();


    FreeWill::ConvolutionDerivative<FreeWill::CPU, float> convolutionDerivative;
    convolutionDerivative.setInputParameter("PrevActivation", &prevActivaion);
    convolutionDerivative.setInputParameter("FeatureMap", &featureMaps);
    convolutionDerivative.setInputParameter("OutputGrad", &outputGrad);

    convolutionDerivative.setOutputParameter("InputGrad", &inputGrad);
    convolutionDerivative.setOutputParameter("FeatureMapGrad", &featureMapGrad);
    convolutionDerivative.setOutputParameter("BiasGrad", &biasGrad);

    QVERIFY(convolutionDerivative.init());

    float inputArray[] = {
            0,1,1, 0,1,1, 2,1,0, 0,1,2, 2,0,2,
            0,1,0,1,0,0,2,0,0,0,0,0,0,1,1,
            0,0,0,0,0,1,2,1,1,2,0,1,1,0,0,
            1,1,1,0,2,0,0,0,2,2,2,0,0,2,1,
            0,1,0,0,0,2,0,1,0,2,2,1,2,0,1
    };

    float featureMapArray[] = {
            0,0,1,-1,1,0,1,1,1,
            0,0,-1,1,-1,-1,0,1,-1,
            0,1,-1,-1,1,0,-1,0,-1,
            
            1,0,-1, 1,0,1, 1,-1,0,
            1,0,-1, 0,0,1, -1,0,1,
            -1,1,-1, -1,0,0, 1,1,-1
    };

    float biasArray[] = {1,0};

    FreeWill::Tensor<FreeWill::CPU, float> input({3,5,5,1});
    input.init();

    FreeWill::Tensor<FreeWill::CPU, float> bias({2});
    bias.init();

    FreeWill::Tensor<FreeWill::CPU, float> output({2,3,3,1});
    output.init();
    
    unsigned int inputSize = input.shape().size();
    for(unsigned int i = 0;i<inputSize;++i)
    {
        prevActivaion[i] = input[i] = inputArray[i];
    }
   
    unsigned int featureMapSize = featureMaps.shape().size();
    for(unsigned int i =0;i<featureMapSize;++i)
    {
        featureMaps[i] = featureMapArray[i];
    }

    unsigned int biasSize = bias.shape().size();
    for(unsigned int i = 0;i<biasSize;++i)
    {
        bias[i] = biasArray[i];
    }

    FreeWill::Convolution<FreeWill::CPU, float> convolution;
    convolution.setInputParameter("Input", &input);
    convolution.setInputParameter("FeatureMap", &featureMaps);
    convolution.setInputParameter("Bias", &bias);

    convolution.setOutputParameter("Output", &output);

    QVERIFY(convolution.init());

    FreeWill::Sigmoid<FreeWill::CPU, float> sigmoid;
    sigmoid.setInputParameter("Input", &output);
    sigmoid.setOutputParameter("Output", &output);

    QVERIFY(sigmoid.init());

    FreeWill::Tensor<FreeWill::CPU, float> flatOutput({output.shape().size(), 1});
    flatOutput.init();

    

    unsigned int outputSize = output.shape().size();
    
    FreeWill::Tensor<FreeWill::CPU, float> label({outputSize, 1});
    label.init();
    label.randomize();

    FreeWill::Tensor<FreeWill::CPU, float> cost({1});
    cost.init();
    const float epsilon = 0.001;

    FreeWill::CrossEntropy<FreeWill::CPU, float> crossEntropy;
    crossEntropy.setInputParameter("Input", &flatOutput);
    crossEntropy.setInputParameter("Label", &label);
    crossEntropy.setOutputParameter("Cost", &cost);

    QVERIFY(crossEntropy.init());

    for(unsigned int i = 0;i<featureMapSize;++i)
    {
        float origin = featureMaps[i];

        float cost_big = 0;

        featureMaps[i] = origin + epsilon;

        convolution.evaluate();

        sigmoid.evaluate();

        for(unsigned int e = 0;e<outputSize;++e)
        {
            flatOutput[e] = output[e];
        }

        crossEntropy.evaluate();

        cost_big = cost[0];

        float cost_small = 0;

        featureMaps[i] = origin - epsilon;

        convolution.evaluate();

        sigmoid.evaluate();

        for(unsigned int e=0;e<outputSize;++e)
        {
            flatOutput[e] = output[e];
        }

        crossEntropy.evaluate();

        cost_small = cost[0];

        float fakeGrad = (cost_big - cost_small) / (2.0*epsilon);

        fakeFeatureMapGrad[i] = fakeGrad;

        featureMaps[i] = origin;
    }

    convolution.evaluate();
    sigmoid.evaluate();

    for(unsigned int e = 0;e<outputSize;++e)
    {
        flatOutput[e] = output[e];
    }

    crossEntropy.evaluate();

    FreeWill::Tensor<FreeWill::CPU, float> flatOuputGrad({outputSize,1});
    flatOuputGrad.init();

    FreeWill::SigmoidCrossEntropyDerivative<FreeWill::CPU, float> sigmoidCrossEntropyDerivative;
    sigmoidCrossEntropyDerivative.setInputParameter("Input", &flatOutput);
    sigmoidCrossEntropyDerivative.setInputParameter("Label", &label);
    sigmoidCrossEntropyDerivative.setOutputParameter("Output", &flatOuputGrad);

    QVERIFY(sigmoidCrossEntropyDerivative.init());

    sigmoidCrossEntropyDerivative.evaluate();

    for(unsigned int e = 0;e<outputSize;++e)
    {
        outputGrad[e] = flatOuputGrad[e];
    }

    convolutionDerivative.evaluate();

   for(unsigned int i = 0;i<featureMapSize;++i)
   {
    qDebug() << "fake" << fakeFeatureMapGrad[i] << "real" << featureMapGrad[i];
   } 
}
