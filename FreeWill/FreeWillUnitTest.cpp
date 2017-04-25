#include "FreeWillUnitTest.h"
#include "Tensor/Tensor.h"
#include "Tensor/ReferenceCountedBlob.h"
#include "Operator/Operator.h"
#include "Operator/ElementwiseAdd.h"
#include "Operator/Activation.h"
#include "Operator/ActivationDerivative.h"
#include "Operator/CrossEntropyLoss.h"
#include "Operator/SigmoidCrossEntropyLossDerivative.h"
#include "Operator/DotProductWithBias.h"
#include "Operator/DotProductWithBiasDerivative.h"
#include "Operator/SoftmaxLogLoss.h"
#include "Operator/SoftmaxLogLossDerivative.h"
#include "Operator/MaxPooling.h"
#include "Operator/MaxPoolingDerivative.h"
#include "Model/Model.h"

void FreeWillUnitTest::operatorSigmoidCrossEntropyTestCPUAndGPU()
{
    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, float> inputCPU({10,64});
    inputCPU.init();
    inputCPU.randomize();

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, float> labelCPU({10, 64});
    labelCPU.init();
    labelCPU.randomize();

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, float> costCPU({1, 64});
    costCPU.init();
    costCPU.randomize();

    FreeWill::CrossEntropyLoss<FreeWill::DeviceType::CPU_NAIVE, float> crossEntropyLossCPU;
    crossEntropyLossCPU.setInputParameter("Input", &inputCPU);
    crossEntropyLossCPU.setInputParameter("Label", &labelCPU);
    crossEntropyLossCPU.setOutputParameter("Cost", &costCPU);

    QVERIFY(crossEntropyLossCPU.init());

    FreeWill::Tensor<FreeWill::DeviceType::GPU_CUDA, float> inputGPU({10,64});
    inputGPU.init();

    for (unsigned int i = 0; i< inputCPU.shape().size();++i)
    {
        if (inputCPU[i] < 0)
        {
            inputCPU[i] = -inputCPU[i];
        }
        inputCPU[i] += 0.1;

        while (inputCPU[i] > 1.0)
        {
            inputCPU[i] -= 1.0;
        }

        inputGPU[i] = inputCPU[i];
    }

    inputGPU.copyFromHostToDevice();

    FreeWill::Tensor<FreeWill::DeviceType::GPU_CUDA, float> labelGPU({10,64});
    labelGPU.init();

    for (unsigned int i = 0; i< labelCPU.shape().size();++i)
    {
        if (labelCPU[i] < 0)
        {
            labelCPU[i] = -labelCPU[i];
        }
        labelCPU[i] += 0.1;

        while(labelCPU[i]>1.0)
        {
            labelCPU[i] -= 1.0;
        }

        labelGPU[i] = labelCPU[i];
    }
    
    labelGPU.copyFromHostToDevice();
    crossEntropyLossCPU.evaluate();


    FreeWill::Tensor<FreeWill::DeviceType::GPU_CUDA, float> costGPU({1, 64});
    costGPU.init();

    FreeWill::CrossEntropyLoss<FreeWill::DeviceType::GPU_CUDA, float> crossEntropyLossGPU;
    crossEntropyLossGPU.setInputParameter("Input", &inputGPU);
    crossEntropyLossGPU.setInputParameter("Label", &labelGPU);
    crossEntropyLossGPU.setOutputParameter("Cost", &costGPU);

    QVERIFY(crossEntropyLossGPU.init());

    crossEntropyLossGPU.evaluate();

    costGPU.copyFromDeviceToHost();

    for (unsigned int i = 0; i< costCPU.shape().size(); ++i)
    {
        QVERIFY(std::abs(costCPU[i] - costGPU[i]) < epsilon);
    }
}

void FreeWillUnitTest::operatorSigmoidCrossEntropyDerivativeTest()
{
    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, double> input({10,64});
    input.init();
    input.randomize();

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, double> label({10,64});
    label.init();
    label.randomize();

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, double> output({10, 64});
    output.init();


    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, double> cost({1, 64});
    cost.init();
    cost.randomize();

    FreeWill::Activation<FreeWill::ActivationMode::SIGMOID, FreeWill::DeviceType::CPU_NAIVE, double> sigmoid;
    sigmoid.setInputParameter("Input", &input);
    sigmoid.setOutputParameter("Output", &output);
    
    QVERIFY(sigmoid.init());

    FreeWill::CrossEntropyLoss<FreeWill::DeviceType::CPU_NAIVE, double> crossEntropyLoss;
    crossEntropyLoss.setInputParameter("Input", &output);
    crossEntropyLoss.setInputParameter("Label", &label);
    crossEntropyLoss.setOutputParameter("Cost", &cost);

    QVERIFY(crossEntropyLoss.init());

    //crossEntropy.evaluate();
    //printf("cost:%f\n",cost[0]);

    unsigned int batchSize = 64;
    unsigned int vectorSize = 10;

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, double> fakeGradient({10,64});
    fakeGradient.init();

    for(unsigned int e = 0;e<batchSize;++e)
    {
        for(unsigned int i = 0;i<vectorSize;++i)
        {
            double cost_larger = 0;
            double original = input[e*vectorSize + i];
            input[e*vectorSize + i] = original + epsilon;

            sigmoid.evaluate();
            crossEntropyLoss.evaluate();
            cost_larger = cost[e];

            input[e*vectorSize + i] = original - epsilon;

            double cost_smaller = 0;
            sigmoid.evaluate();
            crossEntropyLoss.evaluate();

            cost_smaller = cost[e];

            //printf("l:%f, s:%f ,%f\n", cost_larger, cost_smaller, (cost_larger-cost_smaller) / (2.0*epsilon));
            fakeGradient[e*vectorSize + i] = (cost_larger - cost_smaller) / (2.0 * epsilon);

            input[e*vectorSize + i] = original;
        }
    }

    sigmoid.evaluate();

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, double> realGradient({10,64});
    realGradient.init();

    FreeWill::SigmoidCrossEntropyLossDerivative<FreeWill::DeviceType::CPU_NAIVE, double> sigmoidCrossEntropyLossDerivative;
    sigmoidCrossEntropyLossDerivative.setInputParameter("Input", &output);
    sigmoidCrossEntropyLossDerivative.setInputParameter("Label", &label);
    sigmoidCrossEntropyLossDerivative.setOutputParameter("Output", &realGradient);

    QVERIFY(sigmoidCrossEntropyLossDerivative.init());

    sigmoidCrossEntropyLossDerivative.evaluate();
    
    unsigned int size = realGradient.shape().size();
    for(unsigned int i = 0; i<size; ++i)
    {
        /*if (!(std::abs(fakeGradient[i] - realGradient[i]) < epsilon))
        {
            qDebug() << fakeGradient[i] << ";" << realGradient[i];
        }*/
        QVERIFY(relativeError(fakeGradient[i], realGradient[i]) < epsilon);
    }    

}

void FreeWillUnitTest::operatorSigmoidCrossEntropyDerivativeTestGPU()
{
    FreeWill::Tensor<FreeWill::DeviceType::GPU_CUDA, double> input({10,64});
    input.init();
    input.randomize();
    input.copyFromHostToDevice();

    FreeWill::Tensor<FreeWill::DeviceType::GPU_CUDA, double> label({10,64});
    label.init();
    label.randomize();
    label.copyFromHostToDevice();

    FreeWill::Tensor<FreeWill::DeviceType::GPU_CUDA, double> output({10, 64});
    output.init();
    output.copyFromHostToDevice();


    FreeWill::Tensor<FreeWill::DeviceType::GPU_CUDA, double> cost({1, 64});
    cost.init();
    cost.randomize();
    cost.copyFromHostToDevice();

    FreeWill::Activation<FreeWill::ActivationMode::SIGMOID, FreeWill::DeviceType::GPU_CUDA, double> sigmoid;
    sigmoid.setInputParameter("Input", &input);
    sigmoid.setOutputParameter("Output", &output);
    
    QVERIFY(sigmoid.init());

    FreeWill::CrossEntropyLoss<FreeWill::DeviceType::GPU_CUDA, double> crossEntropyLoss;
    crossEntropyLoss.setInputParameter("Input", &output);
    crossEntropyLoss.setInputParameter("Label", &label);
    crossEntropyLoss.setOutputParameter("Cost", &cost);

    QVERIFY(crossEntropyLoss.init());

    //crossEntropy.evaluate();
    //printf("cost:%f\n",cost[0]);

    unsigned int batchSize = 64;
    unsigned int vectorSize = 10;

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, double> fakeGradient({10,64});
    fakeGradient.init();

    for(unsigned int e = 0;e<batchSize;++e)
    {
        for(unsigned int i = 0;i<vectorSize;++i)
        {
            double cost_larger = 0;
            double original = input[e*vectorSize + i];
            input[e*vectorSize + i] = original + epsilon;
            input.copyFromHostToDevice();
            sigmoid.evaluate();
            crossEntropyLoss.evaluate();
            cost.copyFromDeviceToHost();
            cost_larger = cost[e];

            input[e*vectorSize + i] = original - epsilon;
            input.copyFromHostToDevice();
            double cost_smaller = 0;
            sigmoid.evaluate();
            crossEntropyLoss.evaluate();
            cost.copyFromDeviceToHost();
            cost_smaller = cost[e];

            //printf("l:%f, s:%f ,%f\n", cost_larger, cost_smaller, (cost_larger-cost_smaller) / (2.0*epsilon));
            fakeGradient[e*vectorSize + i] = (cost_larger - cost_smaller) / (2.0 * epsilon);

            input[e*vectorSize + i] = original;
            input.copyFromHostToDevice();
        }
    }

    sigmoid.evaluate();

    FreeWill::Tensor<FreeWill::DeviceType::GPU_CUDA, double> realGradient({10,64});
    realGradient.init();

    FreeWill::SigmoidCrossEntropyLossDerivative<FreeWill::DeviceType::GPU_CUDA, double> SigmoidCrossEntropyLossDerivative;
    SigmoidCrossEntropyLossDerivative.setInputParameter("Input", &output);
    SigmoidCrossEntropyLossDerivative.setInputParameter("Label", &label);
    SigmoidCrossEntropyLossDerivative.setOutputParameter("Output", &realGradient);

    QVERIFY(SigmoidCrossEntropyLossDerivative.init());

    SigmoidCrossEntropyLossDerivative.evaluate();
    realGradient.copyFromDeviceToHost();
    
    unsigned int size = realGradient.shape().size();
    for(unsigned int i = 0; i<size; ++i)
    {
        /*if (!(std::abs(fakeGradient[i] - realGradient[i]) < epsilon))
        {
            qDebug() << fakeGradient[i] << ";" << realGradient[i];
        }*/
        QVERIFY(relativeError(fakeGradient[i], realGradient[i]) < epsilon);
    }    

}



void FreeWillUnitTest::operatorDotProductWithBiasTest()
{
    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, float> inputNeurons({4, 2});
    inputNeurons.init({1.1f, 2.1f, 3.1f, 4.1f,
                       5.2f, 6.2f, 7.2f, 8.2f});

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, float> weights({3, 4});
    weights.init({1.3,2.3,3.3,
                   4.3,5.3,6.3,
                   7.3,6.3,9.3,
                   10.3,11.3,12.3});

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, float> bias({3});
    bias.init({13.3,14.3,15.3});

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, float> outputNeurons({3, 2});
    outputNeurons.init();

    FreeWill::DotProductWithBias<FreeWill::DeviceType::CPU_NAIVE, float> dotProductWithBias(true);

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

    for(int i = 0; i<6;++i)
    {
        //qDebug() <<i<< "output neuron" << outputNeurons[i] << "reference" <<reference[i];
        QVERIFY(std::abs(outputNeurons[i] - reference[i]) < epsilon);
    }
}

void FreeWillUnitTest::operatorDotProductWithBiasTestGPU()
{
    FreeWill::Tensor<FreeWill::DeviceType::GPU_CUDA, float> inputNeurons({4, 2});
    inputNeurons.init({1.1f, 2.1f, 3.1f, 4.1f,
                       5.2f, 6.2f, 7.2f, 8.2f});
    inputNeurons.copyFromHostToDevice();

    FreeWill::Tensor<FreeWill::DeviceType::GPU_CUDA, float> weights({3, 4});
    weights.init({1.3,2.3,3.3,
                   4.3,5.3,6.3,
                   7.3,6.3,9.3,
                   10.3,11.3,12.3});
    weights.copyFromHostToDevice();

    FreeWill::Tensor<FreeWill::DeviceType::GPU_CUDA, float> bias({3});
    bias.init({13.3,14.3,15.3});
    bias.copyFromHostToDevice();

    FreeWill::Tensor<FreeWill::DeviceType::GPU_CUDA, float> outputNeurons({3, 2});
    outputNeurons.init();

    FreeWill::DotProductWithBias<FreeWill::DeviceType::GPU_CUDA, float> dotProductWithBias(true);

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

    for(int i = 0; i<6;++i)
    {
       // qDebug() <<i<< "output neuron" << outputNeurons[i] << "reference" <<reference[i];
        QVERIFY(std::abs(outputNeurons[i] - reference[i]) < epsilon);
    }
}

void FreeWillUnitTest::operatorDotProductWithBiasDerivativeTest()
{
    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, double> input({10, 1});
    input.init();
    input.randomize();

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, double> weight({5, 10});
    weight.init();
    weight.randomize();

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, double> bias({5});
    bias.init();
    bias.randomize();

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, double> output({5, 1});
    output.init();

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, double> fakeWeightGrad({5, 10});
    fakeWeightGrad.init();

    FreeWill::DotProductWithBias<FreeWill::DeviceType::CPU_NAIVE, double> dotProductWithBias(true);
    dotProductWithBias.setInputParameter("Input", &input);
    dotProductWithBias.setInputParameter("Weight", &weight);
    dotProductWithBias.setInputParameter("Bias", &bias);
    dotProductWithBias.setOutputParameter("Output", &output);

    QVERIFY(dotProductWithBias.init());

    FreeWill::Activation<FreeWill::ActivationMode::SIGMOID, FreeWill::DeviceType::CPU_NAIVE, double> sigmoid;
    sigmoid.setInputParameter("Input", &output);
    sigmoid.setOutputParameter("Output", &output);

    QVERIFY(sigmoid.init());

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, double> label({5,1});
    label.init();
    label.randomize();

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, double> cost({1,1});
    cost.init();

    FreeWill::CrossEntropyLoss<FreeWill::DeviceType::CPU_NAIVE, double> crossEntropyLoss;
    crossEntropyLoss.setInputParameter("Input", &output);
    crossEntropyLoss.setInputParameter("Label", &label);
    crossEntropyLoss.setOutputParameter("Cost", &cost);

    QVERIFY(crossEntropyLoss.init());

    unsigned int gradientSize = weight.shape().size();

    //Fix me: this doesn't test bias
    for(unsigned int i =0;i<gradientSize;++i)
    {
        double original = weight[i];

        double cost_large = 0;

        weight[i] = original + epsilon;

        dotProductWithBias.evaluate();
        sigmoid.evaluate();
        crossEntropyLoss.evaluate();

        cost_large = cost[0];

        double cost_small = 0;

        weight[i] = original - epsilon;

        dotProductWithBias.evaluate();
        sigmoid.evaluate();
        crossEntropyLoss.evaluate();

        cost_small = cost[0];
        //qDebug() << "large" << cost_large << "small" << cost_small;
        fakeWeightGrad[i] = (cost_large - cost_small) / (2.0 * epsilon);

        weight[i]=original;
    }

    dotProductWithBias.evaluate();
    sigmoid.evaluate();
    crossEntropyLoss.evaluate();

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, double> l1Grad({5,1});
    l1Grad.init();

    FreeWill::SigmoidCrossEntropyLossDerivative<FreeWill::DeviceType::CPU_NAIVE, double> SigmoidCrossEntropyLossDerivative;
    SigmoidCrossEntropyLossDerivative.setInputParameter("Input" , &output);
    SigmoidCrossEntropyLossDerivative.setInputParameter("Label", &label);
    SigmoidCrossEntropyLossDerivative.setOutputParameter("Output", &l1Grad);

    QVERIFY(SigmoidCrossEntropyLossDerivative.init());
    SigmoidCrossEntropyLossDerivative.evaluate();

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, double> realGradient({5,10});
    realGradient.init();

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, double> realBiasGradient({5});
    realBiasGradient.init();

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, double> realInputGradient({10, 1});
    realInputGradient.init();

    FreeWill::DotProductWithBiasDerivative<FreeWill::DeviceType::CPU_NAIVE, double> dotProductWithBiasDerivative(true);
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
        QVERIFY(relativeError(realGradient[i], fakeWeightGrad[i]) < 2.0 * epsilon);
    }
    
}

void FreeWillUnitTest::operatorDotProductWithBiasDerivativeTestGPU()
{
    const unsigned int inputSize = 3;
    const unsigned int outputSize = 2;
    const unsigned int batchSize = 5;

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, double> inputActivationCPU({inputSize, batchSize});
    inputActivationCPU.init();
    inputActivationCPU.randomize();
    

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, double> outputDeltaCPU({outputSize, batchSize});
    outputDeltaCPU.init();
    outputDeltaCPU.randomize();

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, double> weightCPU({outputSize, inputSize});
    weightCPU.init();
    weightCPU.randomize();

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, double> weightGradCPU({outputSize, inputSize});
    weightGradCPU.init();

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, double> biasGradCPU({outputSize});
    biasGradCPU.init();

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, double> inputGradCPU({inputSize, batchSize});
    inputGradCPU.init();
   

    FreeWill::Tensor<FreeWill::DeviceType::GPU_CUDA, double> inputActivationGPU({inputSize, batchSize});
    inputActivationGPU.init();
    double base1 = 0.1;
    for(unsigned int i = 0;i<inputActivationCPU.shape().size();++i)
    {
        inputActivationGPU[i] = (inputActivationCPU[i] = base1);
        base1 += 0.1;
    }
    inputActivationGPU.copyFromHostToDevice();

    FreeWill::Tensor<FreeWill::DeviceType::GPU_CUDA, double> outputDeltaGPU({outputSize,batchSize});
    outputDeltaGPU.init();
    double base2 = 0.2;
    for(unsigned int i =0;i<outputDeltaCPU.shape().size();++i)
    {
        outputDeltaGPU[i] = (outputDeltaCPU[i] = base2);
        base2 += 0.1;
    }
    outputDeltaGPU.copyFromHostToDevice();

    FreeWill::Tensor<FreeWill::DeviceType::GPU_CUDA, double> weightGPU({outputSize,inputSize});
    weightGPU.init();
    for(unsigned int i =0;i<weightCPU.shape().size();++i)
    {
        weightGPU[i] = weightCPU[i];
    }
    weightGPU.copyFromHostToDevice();


    FreeWill::Tensor<FreeWill::DeviceType::GPU_CUDA, double> weightGradGPU({outputSize,inputSize});
    weightGradGPU.init();

    FreeWill::Tensor<FreeWill::DeviceType::GPU_CUDA, double> biasGradGPU({outputSize});
    biasGradGPU.init();

    FreeWill::Tensor<FreeWill::DeviceType::GPU_CUDA, double> inputGradGPU({inputSize,batchSize});
    inputGradGPU.init();

    FreeWill::DotProductWithBiasDerivative<FreeWill::DeviceType::CPU_NAIVE, double> dotProductWithBiasDerivativeCPU;
    dotProductWithBiasDerivativeCPU.setInputParameter("InputActivation", &inputActivationCPU);
    dotProductWithBiasDerivativeCPU.setInputParameter("OutputDelta", &outputDeltaCPU);
    dotProductWithBiasDerivativeCPU.setInputParameter("Weight", &weightCPU);

    dotProductWithBiasDerivativeCPU.setOutputParameter("WeightGrad", &weightGradCPU);
    dotProductWithBiasDerivativeCPU.setOutputParameter("BiasGrad", &biasGradCPU);
    dotProductWithBiasDerivativeCPU.setOutputParameter("InputDelta", &inputGradCPU);

    QVERIFY(dotProductWithBiasDerivativeCPU.init());

    FreeWill::DotProductWithBiasDerivative<FreeWill::DeviceType::GPU_CUDA, double> dotProductWithBiasDerivativeGPU;
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

    const double threshold = 0.01;
    for(unsigned int i=0;i<weightGradCPU.shape().size();++i)
    {
//        qDebug() << "cpu" << weightGradCPU[i] << "gpu" << weightGradGPU[i];
        QVERIFY(std::abs(weightGradCPU[i]- weightGradGPU[i]) < threshold);
    }

    for(unsigned int i=0;i<biasGradCPU.shape().size();++i)
    {
        //qDebug() << "cpu" << biasGradCPU[i] << "gpu" << biasGradGPU[i];
        QVERIFY(std::abs(biasGradCPU[i] - biasGradGPU[i]) < threshold);
    }

    for(unsigned int i=0;i<inputGradGPU.shape().size();++i)
    {
        //qDebug()<<"cpu" << inputGradCPU[i] << "gpu" << inputGradGPU[i];
        QVERIFY(std::abs(inputGradCPU[i] - inputGradGPU[i]) < threshold);
    }

}

void FreeWillUnitTest::SoftmaxTest()
{
    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, float> input({3,1});
    input.init();

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, float> output({3,1});
    output.init();

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, unsigned int> label({1,1});
    label.init();

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, float> cost({1,1});
    cost.init();

    FreeWill::SoftmaxLogLoss<FreeWill::DeviceType::CPU_NAIVE, float> softmaxLogLoss;
    softmaxLogLoss.setInputParameter("Input" , &input);
    softmaxLogLoss.setInputParameter("Label", &label);
    softmaxLogLoss.setOutputParameter("Cost", &cost);
    softmaxLogLoss.setOutputParameter("Output", &output);

    QVERIFY(softmaxLogLoss.init());

    input[0] = -2.85;
    input[1] = 0.86;
    input[2] = 0.28;
    label[0] = 2;

    softmaxLogLoss.evaluate();

    //qDebug() << "softmax cost" << cost[0];
    //
    float groundTruth = 1.04;

    QVERIFY(std::abs(cost[0] - groundTruth) < 0.01);


    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, float> input2({10,1});
    input2.init();

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, float> output2({10,1});
    output2.init();

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, unsigned int> label2({1,1});
    label2.init();

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, float> cost2({1,1});
    cost2.init();

    FreeWill::SoftmaxLogLoss<FreeWill::DeviceType::CPU_NAIVE, float> softmaxLogLoss2;
    softmaxLogLoss2.setInputParameter("Input", &input2);
    softmaxLogLoss2.setInputParameter("Label", &label2);
    softmaxLogLoss2.setOutputParameter("Cost", &cost2);
    softmaxLogLoss2.setOutputParameter("Output", &output2);

    QVERIFY(softmaxLogLoss2.init());

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

    softmaxLogLoss2.evaluate();

    /*for(int i = 0;i<10;++i)
    {
        printf("output:%f\n", output2[i]);
    }

    printf("cost: %f\n", cost2[0]);*/
}


void FreeWillUnitTest::SoftmaxTestGPU()
{
    FreeWill::Tensor<FreeWill::DeviceType::GPU_CUDA, float> input({3,1});
    input.init();

    FreeWill::Tensor<FreeWill::DeviceType::GPU_CUDA, float> output({3,1});
    output.init();

    FreeWill::Tensor<FreeWill::DeviceType::GPU_CUDA, unsigned int> label({1,1});
    label.init();

    FreeWill::Tensor<FreeWill::DeviceType::GPU_CUDA, float> cost({1,1});
    cost.init();

    FreeWill::SoftmaxLogLoss<FreeWill::DeviceType::GPU_CUDA, float> softmaxLogLoss;
    softmaxLogLoss.setInputParameter("Input" , &input);
    softmaxLogLoss.setInputParameter("Label", &label);
    softmaxLogLoss.setOutputParameter("Cost", &cost);
    softmaxLogLoss.setOutputParameter("Output", &output);

    QVERIFY(softmaxLogLoss.init());

    input[0] = -2.85;
    input[1] = 0.86;
    input[2] = 0.28;
    label[0] = 2;
    input.copyFromHostToDevice();
    label.copyFromHostToDevice();
    softmaxLogLoss.evaluate();
    cost.copyFromDeviceToHost();
    output.copyFromDeviceToHost();

    //qDebug() << "softmax cost" << cost[0];
    //
    //float groundTruth = 1.04;

    //QVERIFY(std::abs(cost[0] - groundTruth) < 0.01);


    FreeWill::Tensor<FreeWill::DeviceType::GPU_CUDA, float> input2({10,1});
    input2.init();

    FreeWill::Tensor<FreeWill::DeviceType::GPU_CUDA, float> output2({10,1});
    output2.init();

    FreeWill::Tensor<FreeWill::DeviceType::GPU_CUDA, unsigned int> label2({1,1});
    label2.init();

    FreeWill::Tensor<FreeWill::DeviceType::GPU_CUDA, float> cost2({1,1});
    cost2.init();

    FreeWill::SoftmaxLogLoss<FreeWill::DeviceType::GPU_CUDA, float> softmaxLogLoss2;
    softmaxLogLoss2.setInputParameter("Input", &input2);
    softmaxLogLoss2.setInputParameter("Label", &label2);
    softmaxLogLoss2.setOutputParameter("Cost", &cost2);
    softmaxLogLoss2.setOutputParameter("Output", &output2);

    QVERIFY(softmaxLogLoss2.init());

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

    input2.copyFromHostToDevice();
    label2.copyFromHostToDevice();


    softmaxLogLoss2.evaluate();
    cost2.copyFromDeviceToHost();
    output2.copyFromDeviceToHost();

    /*for(int i = 0;i<10;++i)
    {
        printf("output:%f\n", output2[i]);
    }

    printf("cost: %f\n", cost2[0]);*/
}


void FreeWillUnitTest::SoftmaxDerivativeTest()
{
    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, double> input({3,1});
    input.init();

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, double> fakeGrad({3,1});
    fakeGrad.init();

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, double> output({3,1});
    output.init();

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, unsigned int> label({1,1});
    label.init();

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, double> cost({1,1});
    cost.init();

    FreeWill::SoftmaxLogLoss<FreeWill::DeviceType::CPU_NAIVE, double> softmaxLogLoss;
    softmaxLogLoss.setInputParameter("Input" , &input);
    softmaxLogLoss.setInputParameter("Label", &label);
    softmaxLogLoss.setOutputParameter("Cost", &cost);
    softmaxLogLoss.setOutputParameter("Output", &output);

    QVERIFY(softmaxLogLoss.init());   

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
    
    for(unsigned int e = 0; e<input.shape()[0];++e)
    {
        double original = input[e];

        double cost_large = 0.0;

        input[e] = original + epsilon;
        
        softmaxLogLoss.evaluate();

        cost_large = cost[0];

        cost.clear();
        output.clear();

        double cost_small = 0.0;

        input[e] = original - epsilon;

        softmaxLogLoss.evaluate();

        cost_small = cost[0];

        cost.clear();
        output.clear();

        fakeGrad[e] = (cost_large - cost_small) / (2.0 * epsilon);

        input[e] = original;
    }

    cost.clear();
    output.clear();
    softmaxLogLoss.evaluate();

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, double> inputGrad({3,1});
    inputGrad.init();

    FreeWill::SoftmaxLogLossDerivative<FreeWill::DeviceType::CPU_NAIVE, double> softmaxLogLossDerivative;
    softmaxLogLossDerivative.setInputParameter("Output", &output);
    softmaxLogLossDerivative.setInputParameter("Label", &label);
    softmaxLogLossDerivative.setOutputParameter("InputGrad", &inputGrad);


    QVERIFY(softmaxLogLossDerivative.init());

    softmaxLogLossDerivative.evaluate();

    for(unsigned int i = 0;i<inputGrad.shape()[0];++i)
    {
        //qDebug() << "fake" << fakeGrad[i] << "real" << inputGrad[i];
        QVERIFY(relativeError(fakeGrad[i], inputGrad[i]) < epsilon);
    }





    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, double> input2({10,1});
    input2.init();

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, double> output2({10,1});
    output2.init();

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, unsigned int> label2({1,1});
    label2.init();

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, double> cost2({1,1});
    cost2.init();

    FreeWill::SoftmaxLogLoss<FreeWill::DeviceType::CPU_NAIVE, double> softmaxLogLoss2;
    softmaxLogLoss2.setInputParameter("Input", &input2);
    softmaxLogLoss2.setInputParameter("Label", &label2);
    softmaxLogLoss2.setOutputParameter("Cost", &cost2);
    softmaxLogLoss2.setOutputParameter("Output", &output2);

    QVERIFY(softmaxLogLoss2.init());

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

    input2[0] = -0.149088;
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

    softmaxLogLoss2.evaluate();

 
    
    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, double> inputGrad2({10,1});
    inputGrad2.init();

    FreeWill::SoftmaxLogLossDerivative<FreeWill::DeviceType::CPU_NAIVE, double> softmaxLogLossDerivative2;
    softmaxLogLossDerivative2.setInputParameter("Output", &output2);
    softmaxLogLossDerivative2.setInputParameter("Label", &label2);
    softmaxLogLossDerivative2.setOutputParameter("InputGrad", &inputGrad2);


    QVERIFY(softmaxLogLossDerivative2.init());

    softmaxLogLossDerivative2.evaluate();

    for(unsigned int i = 0;i<inputGrad2.shape()[0];++i)
    {
        //qDebug() << "fake" << fakeGrad[i] << "real" << inputGrad[i];
        //QVERIFY(relativeError(fakeGrad[i], inputGrad[i]) < epsilon);
        //printf("inputgrad: %f\n", inputGrad2[i]);
    }


}

void FreeWillUnitTest::SoftmaxDerivativeTestGPU()
{
    FreeWill::Tensor<FreeWill::DeviceType::GPU_CUDA, double> input({3,1});
    input.init();

    FreeWill::Tensor<FreeWill::DeviceType::GPU_CUDA, double> fakeGrad({3,1});
    fakeGrad.init();

    FreeWill::Tensor<FreeWill::DeviceType::GPU_CUDA, double> output({3,1});
    output.init();

    FreeWill::Tensor<FreeWill::DeviceType::GPU_CUDA, unsigned int> label({1,1});
    label.init();

    FreeWill::Tensor<FreeWill::DeviceType::GPU_CUDA, double> cost({1,1});
    cost.init();

    FreeWill::SoftmaxLogLoss<FreeWill::DeviceType::GPU_CUDA, double> softmaxLogLoss;
    softmaxLogLoss.setInputParameter("Input" , &input);
    softmaxLogLoss.setInputParameter("Label", &label);
    softmaxLogLoss.setOutputParameter("Cost", &cost);
    softmaxLogLoss.setOutputParameter("Output", &output);

    QVERIFY(softmaxLogLoss.init());

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


    for(unsigned int e = 0; e<input.shape()[0];++e)
    {
        double original = input[e];

        double cost_large = 0.0;

        input[e] = original + epsilon;
        input.copyFromHostToDevice();
        label.copyFromHostToDevice();
        softmaxLogLoss.evaluate();
        cost.copyFromDeviceToHost();
        output.copyFromDeviceToHost();

        cost_large = cost[0];

        cost.clear();
        output.clear();

        double cost_small = 0.0;

        input[e] = original - epsilon;

        input.copyFromHostToDevice();
        label.copyFromHostToDevice();
        softmaxLogLoss.evaluate();
        cost.copyFromDeviceToHost();
        output.copyFromDeviceToHost();

        cost_small = cost[0];

        cost.clear();
        output.clear();

        fakeGrad[e] = (cost_large - cost_small) / (2.0 * epsilon);

        input[e] = original;
    }

    cost.clear();
    output.clear();

    input.copyFromHostToDevice();
    label.copyFromHostToDevice();
    softmaxLogLoss.evaluate();
    output.copyFromDeviceToHost();
    cost.copyFromDeviceToHost();

    FreeWill::Tensor<FreeWill::DeviceType::GPU_CUDA, double> inputGrad({3,1});
    inputGrad.init();

    FreeWill::SoftmaxLogLossDerivative<FreeWill::DeviceType::GPU_CUDA, double> softmaxLogLossDerivative;
    softmaxLogLossDerivative.setInputParameter("Output", &output);
    softmaxLogLossDerivative.setInputParameter("Label", &label);
    softmaxLogLossDerivative.setOutputParameter("InputGrad", &inputGrad);


    QVERIFY(softmaxLogLossDerivative.init());

    softmaxLogLossDerivative.evaluate();
    inputGrad.copyFromDeviceToHost();

    for(unsigned int i = 0;i<inputGrad.shape()[0];++i)
    {
        //qDebug() << "fake" << fakeGrad[i] << "real" << inputGrad[i];
        QVERIFY(relativeError(fakeGrad[i], inputGrad[i]) < epsilon);
    }


    FreeWill::Tensor<FreeWill::DeviceType::GPU_CUDA, double> input2({10,1});
    input2.init();

    FreeWill::Tensor<FreeWill::DeviceType::GPU_CUDA, double> output2({10,1});
    output2.init();

    FreeWill::Tensor<FreeWill::DeviceType::GPU_CUDA, unsigned int> label2({1,1});
    label2.init();

    FreeWill::Tensor<FreeWill::DeviceType::GPU_CUDA, double> cost2({1,1});
    cost2.init();

    FreeWill::SoftmaxLogLoss<FreeWill::DeviceType::GPU_CUDA, double> softmaxLogLoss2;
    softmaxLogLoss2.setInputParameter("Input", &input2);
    softmaxLogLoss2.setInputParameter("Label", &label2);
    softmaxLogLoss2.setOutputParameter("Cost", &cost2);
    softmaxLogLoss2.setOutputParameter("Output", &output2);

    QVERIFY(softmaxLogLoss2.init());

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

    input2.copyFromHostToDevice();
    label2.copyFromHostToDevice();
    softmaxLogLoss2.evaluate();



    FreeWill::Tensor<FreeWill::DeviceType::GPU_CUDA, double> inputGrad2({10,1});
    inputGrad2.init();

    FreeWill::SoftmaxLogLossDerivative<FreeWill::DeviceType::GPU_CUDA, double> softmaxLogLossDerivative2;
    softmaxLogLossDerivative2.setInputParameter("Output", &output2);
    softmaxLogLossDerivative2.setInputParameter("Label", &label2);
    softmaxLogLossDerivative2.setOutputParameter("InputGrad", &inputGrad2);


    QVERIFY(softmaxLogLossDerivative2.init());

    softmaxLogLossDerivative2.evaluate();


    for(unsigned int i = 0;i<inputGrad2.shape()[0];++i)
    {
        //qDebug() << "fake" << fakeGrad[i] << "real" << inputGrad[i];
        //QVERIFY(relativeError(fakeGrad[i], inputGrad[i]) < epsilon);
        //printf("inputgrad: %f\n", inputGrad2[i]);
    }


}

void FreeWillUnitTest::maxPoolingTestCPUAndGPU()
{
    FreeWill::Tensor<FreeWill::DeviceType::GPU_CUDA, float> inputGPU({3,10,10,2});
    inputGPU.init();
    inputGPU.randomize();

    FreeWill::Tensor<FreeWill::DeviceType::GPU_CUDA, float> outputGPU({3,5,5,2});
    outputGPU.init();

    FreeWill::Tensor<FreeWill::DeviceType::GPU_CUDA, unsigned int> switchX({3,5,5,2});
    switchX.init();
    
    FreeWill::Tensor<FreeWill::DeviceType::GPU_CUDA, unsigned int> switchY({3,5,5,2});
    switchY.init();


    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, float> input({3,10,10,2});
    input.init();
    
    for (unsigned int i = 0; i<input.shape().size();++i)
    {
        input[i] = inputGPU[i];
    }

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, float> output({3,5,5,2});
    output.init();
    
    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, unsigned int> switchXCPU({3,5,5,2});
    switchXCPU.init();

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, unsigned int> switchYCPU({3,5,5,2});
    switchYCPU.init();


    FreeWill::MaxPooling<FreeWill::DeviceType::GPU_CUDA, float> maxpoolingGPU;
    maxpoolingGPU.setInputParameter("Input", &inputGPU);
    maxpoolingGPU.setOutputParameter("Output", &outputGPU);
    maxpoolingGPU.setOutputParameter("SwitchX", &switchX);
    maxpoolingGPU.setOutputParameter("SwitchY", &switchY);
    QVERIFY(maxpoolingGPU.init());

    FreeWill::MaxPooling<FreeWill::DeviceType::CPU_NAIVE, float> maxpoolingCPU;
    maxpoolingCPU.setInputParameter("Input", &input);
    maxpoolingCPU.setOutputParameter("Output", &output);
    maxpoolingCPU.setOutputParameter("SwitchX", &switchXCPU);
    maxpoolingCPU.setOutputParameter("SwitchY", &switchYCPU);
    QVERIFY(maxpoolingCPU.init());

    inputGPU.copyFromHostToDevice();

    maxpoolingGPU.evaluate();
    maxpoolingCPU.evaluate();

    outputGPU.copyFromDeviceToHost();

    for(unsigned int i =0;i<outputGPU.shape().size(); ++i)
    {
        //qDebug() << outputGPU[i] << output[i];
        QVERIFY(std::abs(outputGPU[i] - output[i]) < epsilon);
    }

    FreeWill::Tensor<FreeWill::DeviceType::GPU_CUDA, float> inputGradGPU({3,10,10,2});
    inputGradGPU.init();

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, float> inputGradCPU({3,10,10,2});
    inputGradCPU.init();

    FreeWill::Tensor<FreeWill::DeviceType::GPU_CUDA, float> outputGradGPU({3,5,5,2});
    outputGradGPU.init();
    outputGradGPU.randomize();

    FreeWill::Tensor<FreeWill::DeviceType::CPU_NAIVE, float> outputGradCPU({3,5,5,2});
    outputGradCPU.init();

    for (unsigned int i = 0;i<outputGradGPU.shape().size();++i)
    {
        outputGradCPU[i] = outputGradGPU[i];
    }

    FreeWill::MaxPoolingDerivative<FreeWill::DeviceType::GPU_CUDA, float> maxPoolingDerivativeGPU;
    maxPoolingDerivativeGPU.setInputParameter("Input", &inputGPU);
    maxPoolingDerivativeGPU.setInputParameter("Output", &outputGPU);
    maxPoolingDerivativeGPU.setInputParameter("OutputGrad", &outputGradGPU);
    maxPoolingDerivativeGPU.setOutputParameter("InputGrad", &inputGradGPU);

    QVERIFY(maxPoolingDerivativeGPU.init());

    FreeWill::MaxPoolingDerivative<FreeWill::DeviceType::CPU_NAIVE, float> maxPoolingDerivativeCPU;
    maxPoolingDerivativeCPU.setInputParameter("OutputGrad", &outputGradCPU);
    maxPoolingDerivativeCPU.setInputParameter("SwitchX", &switchXCPU);
    maxPoolingDerivativeCPU.setInputParameter("SwitchY", &switchYCPU);
    maxPoolingDerivativeCPU.setOutputParameter("InputGrad", &inputGradCPU);

    QVERIFY(maxPoolingDerivativeCPU.init());

    outputGradGPU.copyFromHostToDevice();
    maxPoolingDerivativeCPU.evaluate();
    maxPoolingDerivativeGPU.evaluate();

    inputGradGPU.copyFromDeviceToHost();


    for(unsigned int i = 0; i< inputGradGPU.shape().size();++i)
    {
        //qDebug() << inputGradGPU[i] << inputGradCPU[i];
        QVERIFY(std::abs(inputGradCPU[i] - inputGradGPU[i])<epsilon);

        //qDebug()<< FreeWill::Operator<FreeWill::CPU>::m_operatorFactoryInitializer.getA();
    }
}

QTEST_MAIN(FreeWillUnitTest)
#include "FreeWillUnitTest.moc"
