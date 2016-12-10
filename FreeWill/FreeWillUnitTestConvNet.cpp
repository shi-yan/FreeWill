#include "FreeWillUnitTest.h"
#include "Operator/Convolution.h"
#include "Operator/ConvolutionDerivative.h"
#include "Operator/ReLU.h"
#include "Operator/ReLUDerivative.h"

void FreeWillUnitTest::convNetTest()
{
    FreeWill::Tensor<FreeWill::CPU, float> input({3,227,227,1});
    input.init();
    
    FreeWill::Tensor<FreeWill::CPU, float> featureMaps({3,5,5,20});
    featureMaps.init();

    FreeWill::Tensor<FreeWill::CPU, float> output({20,55,55,1});
    output.init();

    FreeWill::Tensor<FreeWill::CPU, float> bias({20,1});
    bias.init();


    FreeWill::Convolution<FreeWill::CPU, float> convolution;
    convolution.setInputParameter("Input", &input);
    convolution.setInputParameter("FeatureMap", &featureMaps);
    convolution.setInputParameter("Bias", &bias);
    convolution.setOutputParameter("Output", &output);

    convolution.init();


}
