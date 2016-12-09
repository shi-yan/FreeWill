#include "FreeWillUnitTest.h"
#include "Tensor/Tensor.h"
#include "Tensor/ReferenceCountedBlob.h"
#include "Operator/Operator.h"
#include "Operator/ElementwiseAdd.h"
#include "Operator/Sigmoid.h"
#include "Operator/SigmoidDerivative.h"
#include "Operator/CrossEntropy.h"
#include "Operator/SigmoidCrossEntropyDerivative.h"
#include "Operator/DotProductWithBias.h"
#include "Operator/DotProductWithBiasDerivative.h"

void FreeWillUnitTest::operatorSigmoidTest()
{
    FreeWill::Tensor< FreeWill::CPU_NAIVE, float> input({64,0,32,32});
    input.init();
    input.randomize();

    FreeWill::Tensor< FreeWill::CPU_NAIVE, float> output({64,0,32,32});
    output.init();

    FreeWill::Sigmoid< FreeWill::CPU_NAIVE, float> sigmoid;
    sigmoid.setInputParameter("Input", &input);
    sigmoid.setOutputParameter("Output", &output);

    sigmoid.init();
    sigmoid.evaluate();
}

void FreeWillUnitTest::operatorSigmoidDerivativeTest()
{
    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> input({1});
    input.init();
    input.randomize();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> output({1});
    output.init();

    FreeWill::Sigmoid<FreeWill::CPU_NAIVE, float> sigmoid;
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
    

    FreeWill::SigmoidDerivative<FreeWill::CPU_NAIVE, float> sigmoidDerivative;
    sigmoidDerivative.setInputParameter("Input", &output);
    sigmoidDerivative.setOutputParameter("Output", &input);

    QVERIFY(sigmoidDerivative.init());
    sigmoidDerivative.evaluate();

    //qDebug() << "Gradient check for sigmoid:" << fakeDerivative << " (fake), " << input[0] << " (real)";

    QVERIFY(std::abs(input[0] - fakeDerivative) < epsilon);
}

void FreeWillUnitTest::operatorSigmoidCrossEntropyTest()
{
    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> input({10,64});
    input.init();
    input.randomize();

    //FreeWill::Tensor<FreeWill::CPU_NAIVE, float> output({10, 64});
    //output.init();
    //output.randomize();

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

    FreeWill::Sigmoid<FreeWill::CPU_NAIVE, float> sigmoid;
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
        //qDebug() << fakeGradient[i] << ";" << realGradient[i];
        QVERIFY(std::abs(fakeGradient[i] - realGradient[i]) < epsilon);
    }    

}

void FreeWillUnitTest::operatorDotProductWithBiasTest()
{
    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> inputNeurons({4, 2});
    inputNeurons.init({1.1f, 2.1f, 3.1f, 4.1f,
                       5.2f, 6.2f, 7.2f, 8.2f});

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> weights({3, 5});
    weights.init({1.3,2.3,3.3,
                   4.3,5.3,6.3,
                   7.3,6.3,9.3,
                   10.3,11.3,12.3,
                   13.3,14.3,15.3});

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

void FreeWillUnitTest::operatorDotProductWithBiasDerivativeTest()
{
    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> input({10, 1});
    input.init();
    input.randomize();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> weight({5, 11});
    weight.init();
    weight.randomize();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> output({5, 1});
    output.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> fakeWeightGrad({5, 11, 1});
    fakeWeightGrad.init();

    FreeWill::DotProductWithBias<FreeWill::CPU_NAIVE, float> dotProductWithBias(true);
    dotProductWithBias.setInputParameter("Input", &input);
    dotProductWithBias.setInputParameter("Weight", &weight);
    dotProductWithBias.setOutputParameter("Output", &output);

    QVERIFY(dotProductWithBias.init());

    FreeWill::Sigmoid<FreeWill::CPU_NAIVE, float> sigmoid;
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

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> realGradient({5,11,1});
    realGradient.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> realInputGradient({10, 1});
    realInputGradient.init();

    FreeWill::DotProductWithBiasDerivative<FreeWill::CPU_NAIVE, float> dotProductWithBiasDerivative(true);
    dotProductWithBiasDerivative.setInputParameter("PrevActivation", &input);
    dotProductWithBiasDerivative.setInputParameter("OutputGrad", &l1Grad);
    dotProductWithBiasDerivative.setInputParameter("Weight", &weight);

    dotProductWithBiasDerivative.setOutputParameter("WeightGrad", &realGradient);
    dotProductWithBiasDerivative.setOutputParameter("InputGrad", &realInputGradient);

    QVERIFY(dotProductWithBiasDerivative.init());

    dotProductWithBiasDerivative.evaluate();

    for(unsigned int i = 0;i<gradientSize;++i)
    {
        //qDebug() << "realGradient" << realGradient[i] << "fakeWeightGrad" << fakeWeightGrad[i] << i;
        QVERIFY(std::abs(realGradient[i] - fakeWeightGrad[i]) < 2.0 * epsilon);
    }
    
}

QTEST_MAIN(FreeWillUnitTest)
#include "FreeWillUnitTest.moc"
