#include "FreeWillUnitTest.h"
#include "Tensor/Tensor.h"
#include "Tensor/ReferenceCountedBlob.h"
#include "Operator/Operator.h"
#include "Operator/ElementwiseAdd.h"
#include "Operator/Activation.h"
#include "Operator/ActivationDerivative.h"
#include "Operator/CrossEntropy.h"
#include "Operator/SigmoidCrossEntropyDerivative.h"
#include "Operator/DotProductWithBias.h"
#include "Operator/DotProductWithBiasDerivative.h"
#include "Operator/Softmax.h"
#include "Operator/SoftmaxDerivative.h"

void FreeWillUnitTest::operatorSigmoidCrossEntropyTest()
{
    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> input({10,64});
    input.init();
    input.randomize();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> label({10, 64});
    label.init();
    label.randomize();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> cost({64});
    cost.init();
    cost.randomize();

    FreeWill::CrossEntropy<FreeWill::CPU_NAIVE, float> crossEntropy;
    crossEntropy.setInputParameter("Input", &input);
    crossEntropy.setInputParameter("Label", &label);
    crossEntropy.setOutputParameter("Cost", &cost);

    QVERIFY(crossEntropy.init());
    crossEntropy.evaluate();
}

void FreeWillUnitTest::operatorSigmoidCrossEntropyDerivativeTest()
{
    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> input({10,64});
    input.init();
    input.randomize();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> label({10,64});
    label.init();
    label.randomize();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> output({10, 64});
    output.init();


    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> cost({64});
    cost.init();
    cost.randomize();

    FreeWill::Activation<FreeWill::SIGMOID, FreeWill::CPU_NAIVE, float> sigmoid;
    sigmoid.setInputParameter("Input", &input);
    sigmoid.setOutputParameter("Output", &output);
    
    QVERIFY(sigmoid.init());

    const float epsilon = 0.001;
    //const float threshold = 1e-5;

    FreeWill::CrossEntropy<FreeWill::CPU_NAIVE, float> crossEntropy;
    crossEntropy.setInputParameter("Input", &output);
    crossEntropy.setInputParameter("Label", &label);
    crossEntropy.setOutputParameter("Cost", &cost);

    QVERIFY(crossEntropy.init());

    //crossEntropy.evaluate();
    //printf("cost:%f\n",cost[0]);

    unsigned int batchSize = 64;
    unsigned int vectorSize = 10;

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> fakeGradient({10,64});
    fakeGradient.init();

    for(unsigned int e = 0;e<batchSize;++e)
    {
        for(unsigned int i = 0;i<vectorSize;++i)
        {
            float cost_larger = 0;
            float original = input[e*vectorSize + i];
            input[e*vectorSize + i] = original + epsilon;

            sigmoid.evaluate();
            crossEntropy.evaluate();
            cost_larger = cost[e];

            input[e*vectorSize + i] = original - epsilon;

            float cost_smaller = 0;
            sigmoid.evaluate();
            crossEntropy.evaluate();

            cost_smaller = cost[e];

            //printf("l:%f, s:%f ,%f\n", cost_larger, cost_smaller, (cost_larger-cost_smaller) / (2.0*epsilon));
            fakeGradient[e*vectorSize + i] = (cost_larger - cost_smaller) / (2.0 * epsilon);

            input[e*vectorSize + i] = original;
        }
    }

    sigmoid.evaluate();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> realGradient({10,64});
    realGradient.init();

    FreeWill::SigmoidCrossEntropyDerivative<FreeWill::CPU_NAIVE, float> sigmoidCrossEntropyDerivative;
    sigmoidCrossEntropyDerivative.setInputParameter("Input", &output);
    sigmoidCrossEntropyDerivative.setInputParameter("Label", &label);
    sigmoidCrossEntropyDerivative.setOutputParameter("Output", &realGradient);

    QVERIFY(sigmoidCrossEntropyDerivative.init());

    sigmoidCrossEntropyDerivative.evaluate();
    
    unsigned int size = realGradient.shape().size();
    for(unsigned int i = 0; i<size; ++i)
    {
        if (!(std::abs(fakeGradient[i] - realGradient[i]) < epsilon))
        {
            qDebug() << fakeGradient[i] << ";" << realGradient[i];
        }
        QVERIFY(std::abs(fakeGradient[i] - realGradient[i]) < epsilon);
    }    

}

void FreeWillUnitTest::operatorDotProductWithBiasTest()
{
    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> inputNeurons({4, 2});
    inputNeurons.init({1.1f, 2.1f, 3.1f, 4.1f,
                       5.2f, 6.2f, 7.2f, 8.2f});

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> weights({3, 4});
    weights.init({1.3,2.3,3.3,
                   4.3,5.3,6.3,
                   7.3,6.3,9.3,
                   10.3,11.3,12.3});

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> bias({3});
    bias.init({13.3,14.3,15.3});

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> outputNeurons({3, 2});
    outputNeurons.init();

    FreeWill::DotProductWithBias<FreeWill::CPU_NAIVE, float> dotProductWithBias(true);

    float reference[] = {1.3 * 1.1 + 2.1*4.3 + 7.3*3.1 + 4.1*10.3 + 13.3, 
                         1.1*2.3+2.1*5.3+3.1*6.3+4.1*11.3+14.3, 
                         3.3*1.1+6.3*2.1+9.3*3.1+12.3*4.1+15.3,
                         5.2*1.3+6.2*4.3+7.2*7.3+8.2*10.3+13.3, 
                         5.2*2.3+6.2*5.3+7.2*6.3+8.2*11.3+14.3, 
                         5.2*3.3+6.2*6.3+7.2*9.3+8.2*12.3+15.3};

    dotProductWithBias.setInputParameter("Input", &inputNeurons);
    dotProductWithBias.setInputParameter("Weight", &weights);
    dotProductWithBias.setInputParameter("Bias", &bias);
    dotProductWithBias.setOutputParameter("Output", &outputNeurons);

    QVERIFY(dotProductWithBias.init());

    dotProductWithBias.evaluate();

    const float epsilon = 0.001;

    for(int i = 0; i<6;++i)
    {
        //qDebug() <<i<< "output neuron" << outputNeurons[i] << "reference" <<reference[i];
        QVERIFY(std::abs(outputNeurons[i] - reference[i]) < epsilon);
    }
}

void FreeWillUnitTest::operatorDotProductWithBiasTestGPU()
{
    FreeWill::Tensor<FreeWill::GPU_CUDA, float> inputNeurons({4, 2});
    inputNeurons.init({1.1f, 2.1f, 3.1f, 4.1f,
                       5.2f, 6.2f, 7.2f, 8.2f});
    inputNeurons.copyFromHostToDevice();

    FreeWill::Tensor<FreeWill::GPU_CUDA, float> weights({3, 4});
    weights.init({1.3,2.3,3.3,
                   4.3,5.3,6.3,
                   7.3,6.3,9.3,
                   10.3,11.3,12.3});
    weights.copyFromHostToDevice();

    FreeWill::Tensor<FreeWill::GPU_CUDA, float> bias({3});
    bias.init({13.3,14.3,15.3});
    bias.copyFromHostToDevice();

    FreeWill::Tensor<FreeWill::GPU_CUDA, float> outputNeurons({3, 2});
    outputNeurons.init();

    FreeWill::DotProductWithBias<FreeWill::GPU_CUDA, float> dotProductWithBias(true);

    float reference[] = {1.3 * 1.1 + 2.1*4.3 + 7.3*3.1 + 4.1*10.3+13.3, 
                         1.1*2.3+2.1*5.3+3.1*6.3+4.1*11.3 +14.3, 
                         3.3*1.1+6.3*2.1+9.3*3.1+12.3*4.1+15.3,
                         5.2*1.3+6.2*4.3+7.2*7.3+8.2*10.3+13.3, 
                         5.2*2.3+6.2*5.3+7.2*6.3+8.2*11.3+14.3, 
                         5.2*3.3+6.2*6.3+7.2*9.3+8.2*12.3+15.3};

    dotProductWithBias.setInputParameter("Input", &inputNeurons);
    dotProductWithBias.setInputParameter("Weight", &weights);
    dotProductWithBias.setInputParameter("Bias", &bias);
    dotProductWithBias.setOutputParameter("Output", &outputNeurons);

    QVERIFY(dotProductWithBias.init());

    dotProductWithBias.evaluate();

    outputNeurons.copyFromDeviceToHost();

    const float epsilon = 0.001;

    for(int i = 0; i<6;++i)
    {
       // qDebug() <<i<< "output neuron" << outputNeurons[i] << "reference" <<reference[i];
        QVERIFY(std::abs(outputNeurons[i] - reference[i]) < epsilon);
    }
}

void FreeWillUnitTest::operatorDotProductWithBiasDerivativeTest()
{
    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> input({10, 1});
    input.init();
    input.randomize();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> weight({5, 10});
    weight.init();
    weight.randomize();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> bias({5});
    bias.init();
    bias.randomize();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> output({5, 1});
    output.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> fakeWeightGrad({5, 10});
    fakeWeightGrad.init();

    FreeWill::DotProductWithBias<FreeWill::CPU_NAIVE, float> dotProductWithBias(true);
    dotProductWithBias.setInputParameter("Input", &input);
    dotProductWithBias.setInputParameter("Weight", &weight);
    dotProductWithBias.setInputParameter("Bias", &bias);
    dotProductWithBias.setOutputParameter("Output", &output);

    QVERIFY(dotProductWithBias.init());

    FreeWill::Activation<FreeWill::SIGMOID, FreeWill::CPU_NAIVE, float> sigmoid;
    sigmoid.setInputParameter("Input", &output);
    sigmoid.setOutputParameter("Output", &output);

    QVERIFY(sigmoid.init());

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> label({5,1});
    label.init();
    label.randomize();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> cost({1});
    cost.init();

    FreeWill::CrossEntropy<FreeWill::CPU_NAIVE, float> crossEntropy;
    crossEntropy.setInputParameter("Input", &output);
    crossEntropy.setInputParameter("Label", &label);
    crossEntropy.setOutputParameter("Cost", &cost);

    QVERIFY(crossEntropy.init());

    unsigned int gradientSize = weight.shape().size();

    const float epsilon = 0.001;
    //const float threshold = 1e-5;
    //Fix me: this doesn't test bias
    for(unsigned int i =0;i<gradientSize;++i)
    {
        float original = weight[i];

        float cost_large = 0;

        weight[i] = original + epsilon;

        dotProductWithBias.evaluate();
        sigmoid.evaluate();
        crossEntropy.evaluate();

        cost_large = cost[0];

        float cost_small = 0;

        weight[i] = original - epsilon;

        dotProductWithBias.evaluate();
        sigmoid.evaluate();
        crossEntropy.evaluate();

        cost_small = cost[0];
        //qDebug() << "large" << cost_large << "small" << cost_small;
        fakeWeightGrad[i] = (cost_large - cost_small) / (2.0 * epsilon);

        weight[i]=original;
    }

    dotProductWithBias.evaluate();
    sigmoid.evaluate();
    crossEntropy.evaluate();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> l1Grad({5,1});
    l1Grad.init();

    FreeWill::SigmoidCrossEntropyDerivative<FreeWill::CPU_NAIVE, float> sigmoidCrossEntropyDerivative;
    sigmoidCrossEntropyDerivative.setInputParameter("Input" , &output);
    sigmoidCrossEntropyDerivative.setInputParameter("Label", &label);
    sigmoidCrossEntropyDerivative.setOutputParameter("Output", &l1Grad);

    QVERIFY(sigmoidCrossEntropyDerivative.init());
    sigmoidCrossEntropyDerivative.evaluate();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> realGradient({5,10});
    realGradient.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> realBiasGradient({5});
    realBiasGradient.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> realInputGradient({10, 1});
    realInputGradient.init();

    FreeWill::DotProductWithBiasDerivative<FreeWill::CPU_NAIVE, float> dotProductWithBiasDerivative(true);
    dotProductWithBiasDerivative.setInputParameter("InputActivation", &input);
    dotProductWithBiasDerivative.setInputParameter("OutputDelta", &l1Grad);
    dotProductWithBiasDerivative.setInputParameter("Weight", &weight);

    dotProductWithBiasDerivative.setOutputParameter("WeightGrad", &realGradient);
    dotProductWithBiasDerivative.setOutputParameter("BiasGrad", &realBiasGradient);
    dotProductWithBiasDerivative.setOutputParameter("InputDelta", &realInputGradient);

    QVERIFY(dotProductWithBiasDerivative.init());

    dotProductWithBiasDerivative.evaluate();

    for(unsigned int i = 0;i<gradientSize;++i)
    {
        //qDebug() << "realGradient" << realGradient[i] << "fakeWeightGrad" << fakeWeightGrad[i] << i;
        QVERIFY(std::abs(realGradient[i] - fakeWeightGrad[i]) < 2.0 * epsilon);
    }
    
}

void FreeWillUnitTest::operatorDotProductWithBiasDerivativeTestGPU()
{
    const unsigned int inputSize = 3;
    const unsigned int outputSize = 2;
    const unsigned int batchSize = 5;

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> inputActivationCPU({inputSize, batchSize});
    inputActivationCPU.init();
    inputActivationCPU.randomize();
    

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> outputDeltaCPU({outputSize, batchSize});
    outputDeltaCPU.init();
    outputDeltaCPU.randomize();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> weightCPU({outputSize, inputSize});
    weightCPU.init();
    weightCPU.randomize();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> weightGradCPU({outputSize, inputSize});
    weightGradCPU.init();
    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> biasGradCPU({outputSize});
    biasGradCPU.init();
    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> inputGradCPU({inputSize, batchSize});
    inputGradCPU.init();
   

    FreeWill::Tensor<FreeWill::GPU_CUDA, float> inputActivationGPU({inputSize, batchSize});
    inputActivationGPU.init();
    float base1 = 0.1;
    for(unsigned int i = 0;i<inputActivationCPU.shape().size();++i)
    {
        inputActivationGPU[i] = (inputActivationCPU[i] = base1);
        base1 += 0.1;
    }
    inputActivationGPU.copyFromHostToDevice();

    FreeWill::Tensor<FreeWill::GPU_CUDA, float> outputDeltaGPU({outputSize,batchSize});
    outputDeltaGPU.init();
    float base2 = 0.2;
    for(unsigned int i =0;i<outputDeltaCPU.shape().size();++i)
    {
        outputDeltaGPU[i] = (outputDeltaCPU[i] = base2);
        base2 += 0.1;
    }
    outputDeltaGPU.copyFromHostToDevice();

    FreeWill::Tensor<FreeWill::GPU_CUDA, float> weightGPU({outputSize,inputSize});
    weightGPU.init();
    for(unsigned int i =0;i<weightCPU.shape().size();++i)
    {
        weightGPU[i] = weightCPU[i];
    }
    weightGPU.copyFromHostToDevice();


    FreeWill::Tensor<FreeWill::GPU_CUDA, float> weightGradGPU({outputSize,inputSize});
    weightGradGPU.init();

    FreeWill::Tensor<FreeWill::GPU_CUDA, float> biasGradGPU({outputSize});
    biasGradGPU.init();

    FreeWill::Tensor<FreeWill::GPU_CUDA, float> inputGradGPU({inputSize,batchSize});
    inputGradGPU.init();

    FreeWill::DotProductWithBiasDerivative<FreeWill::CPU_NAIVE, float> dotProductWithBiasDerivativeCPU;
    dotProductWithBiasDerivativeCPU.setInputParameter("InputActivation", &inputActivationCPU);
    dotProductWithBiasDerivativeCPU.setInputParameter("OutputDelta", &outputDeltaCPU);
    dotProductWithBiasDerivativeCPU.setInputParameter("Weight", &weightCPU);

    dotProductWithBiasDerivativeCPU.setOutputParameter("WeightGrad", &weightGradCPU);
    dotProductWithBiasDerivativeCPU.setOutputParameter("BiasGrad", &biasGradCPU);
    dotProductWithBiasDerivativeCPU.setOutputParameter("InputDelta", &inputGradCPU);

    QVERIFY(dotProductWithBiasDerivativeCPU.init());

    FreeWill::DotProductWithBiasDerivative<FreeWill::GPU_CUDA, float> dotProductWithBiasDerivativeGPU;
    dotProductWithBiasDerivativeGPU.setInputParameter("InputActivation", &inputActivationGPU);
    dotProductWithBiasDerivativeGPU.setInputParameter("OutputDelta", &outputDeltaGPU);
    dotProductWithBiasDerivativeGPU.setInputParameter("Weight", &weightGPU);

    dotProductWithBiasDerivativeGPU.setOutputParameter("WeightGrad", &weightGradGPU);
    dotProductWithBiasDerivativeGPU.setOutputParameter("BiasGrad", &biasGradGPU);
    dotProductWithBiasDerivativeGPU.setOutputParameter("InputDelta", &inputGradGPU);

    QVERIFY(dotProductWithBiasDerivativeGPU.init());


    dotProductWithBiasDerivativeCPU.evaluate();
    dotProductWithBiasDerivativeGPU.evaluate();

    weightGradGPU.copyFromDeviceToHost();
    biasGradGPU.copyFromDeviceToHost();
    inputGradGPU.copyFromDeviceToHost();

    float const epsilon = 0.01;
    for(unsigned int i=0;i<weightGradCPU.shape().size();++i)
    {
//        qDebug() << "cpu" << weightGradCPU[i] << "gpu" << weightGradGPU[i];
        QVERIFY(std::abs(weightGradCPU[i] - weightGradGPU[i]) < epsilon);
    }

    for(unsigned int i=0;i<biasGradCPU.shape().size();++i)
    {
        //qDebug() << "cpu" << biasGradCPU[i] <<outputDeltaCPU[i] << "gpu" << biasGradGPU[i];
        QVERIFY(std::abs(biasGradCPU[i] - biasGradGPU[i]) < epsilon);
    }

    for(unsigned int i=0;i<inputGradGPU.shape().size();++i)
    {
        //qDebug()<<"cpu" << inputGradCPU[i] << "gpu" << inputGradGPU[i];
        QVERIFY(std::abs(inputGradCPU[i] - inputGradGPU[i]) < epsilon);
    }

}

void FreeWillUnitTest::SoftmaxTest()
{
    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> input({3,1});
    input.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> output({3,1});
    output.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, unsigned int> label({1});
    label.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> cost({1});
    cost.init();

    FreeWill::Softmax<FreeWill::CPU_NAIVE, float> softmax;
    softmax.setInputParameter("Input" , &input);
    softmax.setInputParameter("Label", &label);
    softmax.setOutputParameter("Cost", &cost);
    softmax.setOutputParameter("Output", &output);

    QVERIFY(softmax.init());   

    input[0] = -2.85;
    input[1] = 0.86;
    input[2] = 0.28;
    label[0] = 2;

    softmax.evaluate();
    
    //qDebug() << "softmax cost" << cost[0];
    //
    float groundTruth = 1.04;

    QVERIFY(std::abs(cost[0] - groundTruth) < 0.01); 


    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> input2({10,1});
    input2.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> output2({10,1});
    output2.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, unsigned int> label2({1});
    label2.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> cost2({1});
    cost2.init();

    FreeWill::Softmax<FreeWill::CPU_NAIVE, float> softmax2;
    softmax2.setInputParameter("Input", &input2);
    softmax2.setInputParameter("Label", &label2);
    softmax2.setOutputParameter("Cost", &cost2);
    softmax2.setOutputParameter("Output", &output2);

    QVERIFY(softmax2.init());

    input2[0] = 0.116663;
    input2[1] = -0.316017;
    input2[2] = -0.242819;
    input2[3] = -0.157871;
    input2[4] = -0.547314;
    input2[5] = 0.177335;
    input2[6] = -0.101721;
    input2[7] =-0.132597;
    input2[8] = -0.659628;
    input2[9] = 0.697892;

    label2[0] = 5.0;

    softmax2.evaluate();

/*    for(int i = 0;i<10;++i)
    {
        printf("output:%f\n", output2[i]);
    }

    printf("cost: %f\n", cost2[0]);*/
}

void FreeWillUnitTest::SoftmaxDerivativeTest()
{
    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> input({3,1});
    input.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> fakeGrad({3,1});
    fakeGrad.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> output({3,1});
    output.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, unsigned int> label({1});
    label.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> cost({1});
    cost.init();

    FreeWill::Softmax<FreeWill::CPU_NAIVE, float> softmax;
    softmax.setInputParameter("Input" , &input);
    softmax.setInputParameter("Label", &label);
    softmax.setOutputParameter("Cost", &cost);
    softmax.setOutputParameter("Output", &output);

    QVERIFY(softmax.init());   

    input[0] = -2.85;
    input[1] = 0.86;
    input[2] = 0.28;
    label[0] = 2;

    //softmax.evaluate();
    
    //qDebug() << "softmax cost" << cost[0];
    //
    //float groundTruth = 1.04;

    //QVERIFY(std::abs(cost[0] - groundTruth) < 0.01); 
    //
    
    const float epsilon = 0.001;

    for(unsigned int e = 0; e<input.shape()[0];++e)
    {
        float original = input[e];

        float cost_large = 0.0;

        input[e] = original + epsilon;
        
        softmax.evaluate();

        cost_large = cost[0];

        cost.clear();
        output.clear();

        float cost_small = 0.0;

        input[e] = original - epsilon;

        softmax.evaluate();

        cost_small = cost[0];

        cost.clear();
        output.clear();

        fakeGrad[e] = (cost_large - cost_small) / (2.0 * epsilon);

        input[e] = original;
    }

    cost.clear();
    output.clear();
    softmax.evaluate();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> inputGrad({3,1});
    inputGrad.init();

    FreeWill::SoftmaxDerivative<FreeWill::CPU_NAIVE, float> softmaxDerivative;
    softmaxDerivative.setInputParameter("Output", &output);
    softmaxDerivative.setInputParameter("Label", &label);
    softmaxDerivative.setOutputParameter("InputGrad", &inputGrad);


    QVERIFY(softmaxDerivative.init());

    softmaxDerivative.evaluate();

    for(unsigned int i = 0;i<inputGrad.shape()[0];++i)
    {
        //qDebug() << "fake" << fakeGrad[i] << "real" << inputGrad[i];
        QVERIFY((fakeGrad[i] - inputGrad[i]) < epsilon);
    }





    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> input2({10,1});
    input2.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> output2({10,1});
    output2.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, unsigned int> label2({1});
    label2.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> cost2({1});
    cost2.init();

    FreeWill::Softmax<FreeWill::CPU_NAIVE, float> softmax2;
    softmax2.setInputParameter("Input", &input2);
    softmax2.setInputParameter("Label", &label2);
    softmax2.setOutputParameter("Cost", &cost2);
    softmax2.setOutputParameter("Output", &output2);

    QVERIFY(softmax2.init());

/*    input2[0] = 0.116663;
    input2[1] = -0.316017;
    input2[2] = -0.242819;
    input2[3] = -0.157871;
    input2[4] = -0.547314;
    input2[5] = 0.177335;
    input2[6] = -0.101721;
    input2[7] =-0.132597;
    input2[8] = -0.659628;
    input2[9] = 0.697892;
*/

    input2[0]=-0.149088;
    input2[1] = 0.565349;
    input2[2] = -0.733031;
    input2[3] = 0.039112;
    input2[4] = -0.556532;
    input2[5] = -0.009531;
    input2[6] = -0.230422;
    input2[7] = 0.295921;
    input2[8] = 0.535369;
    input2[9] = -0.333607;

    label2[0] = 5.0;

    softmax2.evaluate();

 
    
    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> inputGrad2({10,1});
    inputGrad2.init();

    FreeWill::SoftmaxDerivative<FreeWill::CPU_NAIVE, float> softmaxDerivative2;
    softmaxDerivative2.setInputParameter("Output", &output2);
    softmaxDerivative2.setInputParameter("Label", &label2);
    softmaxDerivative2.setOutputParameter("InputGrad", &inputGrad2);


    QVERIFY(softmaxDerivative2.init());

    softmaxDerivative2.evaluate();
/*
    for(unsigned int i = 0;i<inputGrad2.shape()[0];++i)
    {
        //qDebug() << "fake" << fakeGrad[i] << "real" << inputGrad[i];
//        QVERIFY((fakeGrad[i] - inputGrad[i]) < epsilon);
        printf("inputgrad: %f\n", inputGrad2[i]);
    }

*/
}

QTEST_MAIN(FreeWillUnitTest)
#include "FreeWillUnitTest.moc"
