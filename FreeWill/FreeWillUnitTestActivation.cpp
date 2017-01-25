#include "Tensor/Tensor.h"
#include "Tensor/ReferenceCountedBlob.h"
#include "Operator/Operator.h"
#include "Operator/ElementwiseAdd.h"
#include "Operator/Activation.h"
#include "Operator/ActivationDerivative.h"
#include "FreeWillUnitTest.h"

void FreeWillUnitTest::operatorSigmoidTestCPUAndGPU()
{
    FreeWill::Tensor< FreeWill::CPU_NAIVE, float> inputCPU({64,32,32});
    inputCPU.init();
    inputCPU.randomize();

    FreeWill::Tensor< FreeWill::CPU_NAIVE, float> outputCPU({64,32,32});
    outputCPU.init();

    FreeWill::Activation<FreeWill::SIGMOID, FreeWill::CPU_NAIVE, float> sigmoidCPU;
    sigmoidCPU.setInputParameter("Input", &inputCPU);
    sigmoidCPU.setOutputParameter("Output", &outputCPU);

    QVERIFY(sigmoidCPU.init());
    sigmoidCPU.evaluate();


    FreeWill::Tensor<FreeWill::GPU_CUDA, float> inputGPU({64,32,32});
    inputGPU.init();
    
    for(unsigned int i = 0;i<inputCPU.shape().size();++i)
    {
        inputGPU[i] = inputCPU[i];
    }

    FreeWill::Tensor<FreeWill::GPU_CUDA, float> outputGPU({64,32,32});
    outputGPU.init();

    inputGPU.copyFromHostToDevice();

    FreeWill::Activation<FreeWill::SIGMOID, FreeWill::GPU_CUDA, float> sigmoidGPU;
    sigmoidGPU.setInputParameter("Input", &inputGPU);
    sigmoidGPU.setOutputParameter("Output", &outputGPU);

    QVERIFY(sigmoidGPU.init());
    sigmoidGPU.evaluate();

    outputGPU.copyFromDeviceToHost();

    const float epsilon = 0.01;
    for (unsigned int i = 0;i<inputGPU.shape().size(); ++i)
    {
        QVERIFY(std::abs(outputGPU[i] - outputCPU[i]) < epsilon);
    }
}

void FreeWillUnitTest::operatorSigmoidDerivativeTest()
{
    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> input({1});
    input.init();
    input.randomize();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> output({1});
    output.init();

    FreeWill::Activation<FreeWill::SIGMOID, FreeWill::CPU_NAIVE, float> sigmoid;
    sigmoid.setInputParameter("Input", &input);
    sigmoid.setOutputParameter("Output", &output);
    QVERIFY(sigmoid.init());
    sigmoid.evaluate();

    const float epsilon = 0.001;
    //const float threshold = 1e-5; 

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> input_larger({1});
    input_larger.init();
    input_larger[0] = input[0] + epsilon;
    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> output_larger({1});
    output_larger.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> input_smaller({1});
    input_smaller.init();
    input_smaller[0] = input[0] - epsilon;

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> output_smaller({1});
    output_smaller.init();
    
    sigmoid.clear(); 
    sigmoid.setInputParameter("Input", &input_larger);
    sigmoid.setOutputParameter("Output", &output_larger);

    QVERIFY(sigmoid.init());
    sigmoid.evaluate();

    sigmoid.clear();
    sigmoid.setInputParameter("Input", &input_smaller);
    sigmoid.setOutputParameter("Output", &output_smaller);

    QVERIFY(sigmoid.init());
    sigmoid.evaluate();

    float fakeDerivative = (output_larger[0] - output_smaller[0]) / (2.0 * epsilon); 
   
    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> ones({1});
    ones.init();
    ones[0] = 1; 

    FreeWill::ActivationDerivative<FreeWill::SIGMOID, FreeWill::CPU_NAIVE, float> sigmoidDerivative;
    sigmoidDerivative.setInputParameter("Output", &output);
    sigmoidDerivative.setInputParameter("OutputDelta", &ones);
    sigmoidDerivative.setOutputParameter("InputDelta", &input);

    QVERIFY(sigmoidDerivative.init());
    sigmoidDerivative.evaluate();

    QVERIFY(std::abs(input[0] - fakeDerivative) < epsilon);
}

void FreeWillUnitTest::operatorSigmoidDerivativeTestGPU()
{
    FreeWill::Tensor<FreeWill::GPU_CUDA, float> input({1});
    input.init();
    input.randomize();

    FreeWill::Tensor<FreeWill::GPU_CUDA, float> output({1});
    output.init();

    input.copyFromHostToDevice();

    FreeWill::Activation<FreeWill::SIGMOID, FreeWill::GPU_CUDA, float> sigmoid;
    sigmoid.setInputParameter("Input", &input);
    sigmoid.setOutputParameter("Output", &output);
    QVERIFY(sigmoid.init());
    sigmoid.evaluate();

    output.copyFromDeviceToHost();

    const float epsilon = 0.001;
    //const float threshold = 1e-5; 

    FreeWill::Tensor<FreeWill::GPU_CUDA, float> input_larger({1});
    input_larger.init();
    input_larger[0] = input[0] + epsilon;
    FreeWill::Tensor<FreeWill::GPU_CUDA, float> output_larger({1});
    output_larger.init();

    input_larger.copyFromHostToDevice();

    FreeWill::Tensor<FreeWill::GPU_CUDA, float> input_smaller({1});
    input_smaller.init();
    input_smaller[0] = input[0] - epsilon;

    FreeWill::Tensor<FreeWill::GPU_CUDA, float> output_smaller({1});
    output_smaller.init();

    input_smaller.copyFromHostToDevice();
    
    sigmoid.clear(); 
    sigmoid.setInputParameter("Input", &input_larger);
    sigmoid.setOutputParameter("Output", &output_larger);

    QVERIFY(sigmoid.init());
    sigmoid.evaluate();

    output_larger.copyFromDeviceToHost();

    sigmoid.clear();
    sigmoid.setInputParameter("Input", &input_smaller);
    sigmoid.setOutputParameter("Output", &output_smaller);

    QVERIFY(sigmoid.init());
    sigmoid.evaluate();

    output_smaller.copyFromDeviceToHost();

    float fakeDerivative = (output_larger[0] - output_smaller[0]) / (2.0 * epsilon); 
    

    FreeWill::ActivationDerivative<FreeWill::SIGMOID, FreeWill::GPU_CUDA, float> sigmoidDerivative;
    input[0] = 0;
    input.copyFromHostToDevice();

    FreeWill::Tensor<FreeWill::GPU_CUDA, float> ones({1});
    ones.init();
    ones[0] = 1;
    ones.copyFromHostToDevice();

    sigmoidDerivative.setInputParameter("Output", &output);
    sigmoidDerivative.setInputParameter("OutputDelta", &ones);
    sigmoidDerivative.setOutputParameter("InputDelta", &input);

    QVERIFY(sigmoidDerivative.init());
    sigmoidDerivative.evaluate();

    input.copyFromDeviceToHost();

    printf("fake:%f, real:%f\n",fakeDerivative, input[0]);
    QVERIFY(std::abs(input[0] - fakeDerivative) < epsilon);
}

void FreeWillUnitTest::operatorReLUDerivativeTest()
{
    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> input({1});
    input.init();
    input.randomize();
    input[0] = input[0] - 0.5;

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> output({1});
    output.init();

    FreeWill::Activation<FreeWill::RELU, FreeWill::CPU_NAIVE, float> relu;
    relu.setInputParameter("Input", &input);
    relu.setOutputParameter("Output", &output);
    QVERIFY(relu.init());
    relu.evaluate();

    const float epsilon = 0.001;
    //const float threshold = 1e-5; 

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> input_larger({1});
    input_larger.init();
    input_larger[0] = input[0] + epsilon;
    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> output_larger({1});
    output_larger.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> input_smaller({1});
    input_smaller.init();
    input_smaller[0] = input[0] - epsilon;

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> output_smaller({1});
    output_smaller.init();
    
    relu.clear(); 
    relu.setInputParameter("Input", &input_larger);
    relu.setOutputParameter("Output", &output_larger);

    QVERIFY(relu.init());
    relu.evaluate();

    relu.clear();
    relu.setInputParameter("Input", &input_smaller);
    relu.setOutputParameter("Output", &output_smaller);

    QVERIFY(relu.init());
    relu.evaluate();

    float fakeDerivative = (output_larger[0] - output_smaller[0]) / (2.0 * epsilon); 
   
    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> ones({1});
    ones.init();
    ones[0] = 1; 

    FreeWill::ActivationDerivative<FreeWill::RELU, FreeWill::CPU_NAIVE, float> reluDerivative;
    reluDerivative.setInputParameter("Output", &output);
    reluDerivative.setInputParameter("OutputDelta", &ones);
    reluDerivative.setOutputParameter("InputDelta", &input);

    QVERIFY(reluDerivative.init());
    reluDerivative.evaluate();

    printf("fake %f, real %f\n", fakeDerivative, input[0]);
    QVERIFY(std::abs(input[0] - fakeDerivative) < epsilon);
}


