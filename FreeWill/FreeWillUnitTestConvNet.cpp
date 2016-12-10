#include "FreeWillUnitTest.h"
#include "Operator/Convolution.h"
#include "Operator/ConvolutionDerivative.h"
#include "Operator/ReLU.h"
#include "Operator/ReLUDerivative.h"

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
