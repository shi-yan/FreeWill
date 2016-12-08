#include "FreeWillUnitTest.h"
#include "Tensor/Tensor.h"
#include "Operator/DotProductWithBias.h"
#include "Operator/DotProductWithBiasDerivative.h"
#include "Operator/Sigmoid.h"
#include "Operator/SigmoidDerivative.h"
#include "Operator/CrossEntropy.h"
#include "Operator/SigmoidCrossEntropyDerivative.h"
#include "Operator/ElementwiseAdd.h"
#include "Operator/ElementwiseProduct.h"


void FreeWillUnitTest::xorTest()
{
    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> input({2,1});
    input.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> firstLayerActivation({2 ,1});
    firstLayerActivation.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> firstLayerNeuronDerivative({2, 1});
    firstLayerNeuronDerivative.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> secondLayerActivation({1,1});
    secondLayerActivation.init();
   
    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> secondLayerNeuronDerivative({1,1});
    secondLayerNeuronDerivative.init(); 
    
    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> cost({1});
    cost.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> label({1, 1});
    label.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> firstLayerWeight({2,3});
    firstLayerWeight.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> firstLayerWeightDerivative({2,3,1});
    firstLayerWeightDerivative.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> secondLayerWeight({1, 3});
    secondLayerWeight.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> secondLayerWeightDerivative({1,3,1});
    secondLayerWeightDerivative.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> inputNeuronDerivative({2,1});
    inputNeuronDerivative.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> firstLayerSigmoidNeuron({2,1});
    firstLayerSigmoidNeuron.init();
    
    FreeWill::DotProductWithBias<FreeWill::CPU_NAIVE, float> firstLayerFullyConnected(true);
    firstLayerFullyConnected.setInputParameter("Input", &input);
    firstLayerFullyConnected.setInputParameter("Weight", &firstLayerWeight);
    firstLayerFullyConnected.setOutputParameter("Output", &firstLayerActivation);
    
    QVERIFY(firstLayerActivation.init());


    FreeWill::Sigmoid<FreeWill::CPU_NAIVE, float> firstLayerSigmoid;
    firstLayerSigmoid.setInputParameter("Input", &firstLayerActivation);
    firstLayerSigmoid.setOutputParameter("Output", &firstLayerActivation);
    
    QVERIFY(firstLayerSigmoid.init());

    FreeWill::DotProductWithBias<FreeWill::CPU_NAIVE, float> secondLayerFullyConnected(true);
    secondLayerFullyConnected.setInputParameter("Input", &firstLayerActivation);
    secondLayerFullyConnected.setInputParameter("Weight", &secondLayerWeight);
    secondLayerFullyConnected.setOutputParameter("Output", &secondLayerActivation);

    QVERIFY(secondLayerFullyConnected.init());

    FreeWill::Sigmoid<FreeWill::CPU_NAIVE, float> secondLayerSigmoid;
    secondLayerSigmoid.setInputParameter("Input", &secondLayerActivation);
    secondLayerSigmoid.setOutputParameter("Output", &secondLayerActivation);

    QVERIFY(secondLayerSigmoid.init());

    FreeWill::CrossEntropy<FreeWill::CPU_NAIVE, float> crossEntropy;
    crossEntropy.setInputParameter("Input", &secondLayerActivation);
    crossEntropy.setInputParameter("Label", &label);
    crossEntropy.setOutputParameter("Cost", &cost);

    QVERIFY(crossEntropy.init());


    FreeWill::SigmoidCrossEntropyDerivative<FreeWill::CPU_NAIVE, float> sigmoidCrossEntropyDerivative;
    sigmoidCrossEntropyDerivative.setInputParameter("Input", &secondLayerActivation);
    sigmoidCrossEntropyDerivative.setInputParameter("Label", &label);
    sigmoidCrossEntropyDerivative.setOutputParameter("Output", &secondLayerNeuronDerivative);

    QVERIFY(sigmoidCrossEntropyDerivative.init());

    
    FreeWill::DotProductWithBiasDerivative<FreeWill::CPU_NAIVE, float> secondLayerDotProductWithBiasDerivative(true);
    secondLayerDotProductWithBiasDerivative.setInputParameter("PrevActivation", &firstLayerActivation);
    secondLayerDotProductWithBiasDerivative.setInputParameter("OutputGrad", &secondLayerNeuronDerivative);
    secondLayerDotProductWithBiasDerivative.setInputParameter("Weight", &secondLayerWeight);

    secondLayerDotProductWithBiasDerivative.setOutputParameter("WeightGrad", &secondLayerWeightDerivative);
    secondLayerDotProductWithBiasDerivative.setOutputParameter("InputGrad", &firstLayerNeuronDerivative);

    QVERIFY(secondLayerDotProductWithBiasDerivative.init());

    FreeWill::SigmoidDerivative<FreeWill::CPU_NAIVE, float> firstLayerSigmoidDerivative;
    firstLayerSigmoidDerivative.setInputParameter("Input", &firstLayerActivation);
    firstLayerSigmoidDerivative.setOutputParameter("Output", &firstLayerSigmoidNeuron);

    QVERIFY(firstLayerSigmoidDerivative.init());

    
    FreeWill::ElementwiseProduct<FreeWill::CPU_NAIVE, float> firstLayerDerivativeTimesSigmoidDerivitive;
    firstLayerDerivativeTimesSigmoidDerivitive.setInputParameter("OperandA", &firstLayerSigmoidNeuron);
    firstLayerDerivativeTimesSigmoidDerivitive.setInputParameter("OperandB", &firstLayerNeuronDerivative);
    firstLayerDerivativeTimesSigmoidDerivitive.setOutputParameter("Output", &firstLayerNeuronDerivative);

    QVERIFY(firstLayerDerivativeTimesSigmoidDerivitive.init());
    


    FreeWill::DotProductWithBiasDerivative<FreeWill::CPU_NAIVE, float> firstLayerDotProductWithBiasDerivative(true);
    firstLayerDotProductWithBiasDerivative.setInputParameter("PrevActivation", &input);
    firstLayerDotProductWithBiasDerivative.setInputParameter("OutputGrad", &firstLayerNeuronDerivative);
    firstLayerDotProductWithBiasDerivative.setInputParameter("Weight", &firstLayerWeight);

    firstLayerDotProductWithBiasDerivative.setOutputParameter("WeightGrad", &firstLayerWeightDerivative);
    firstLayerDotProductWithBiasDerivative.setOutputParameter("InputGrad", &inputNeuronDerivative);
     
    FreeWill::ElementwiseAdd<FreeWill::CPU_NAIVE, float> mergeGradWithFirstLayer;
    mergeGradWithFirstLayer.setInputParameter("Operand", &firstLayerWeightDerivative);
    mergeGradWithFirstLayer.setInputParameter("Operand", &firstLayerWeight);
    mergeGradWithFirstLayer.setOutputParameter("Result", &firstLayerWeight);
    QVERIFY(mergeGradWithFirstLayer.init());

    FreeWill::ElementwiseAdd<FreeWill::CPU_NAIVE, float> mergeGradWithSecondLayer;
    mergeGradWithSecondLayer.setInputParameter("Operand", &secondLayerWeightDerivative);
    mergeGradWithSecondLayer.setInputParameter("Operand", &secondLayerWeight);
    mergeGradWithSecondLayer.setOutputParameter("Result", &secondLayerWeight);
    QVERIFY(mergeGradWithSecondLayer.init());




    for(int i = 0; i< 100000; ++i)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis(0, 1);
        
        int a = dis(gen);
        int b = dis(gen);
        int c = a ^ b;

        input[0] = a;
        input[1] = b;
        label[0] = c;

        firstLayerFullyConnected.evaluate();
        firstLayerSigmoid.evaluate();

        secondLayerFullyConnected.evaluate();
        secondLayerSigmoid.evaluate();
        crossEntropy.evaluate();


        qDebug() << "cost" << cost[0];

        sigmoidCrossEntropyDerivative.evaluate();
        secondLayerDotProductWithBiasDerivative.evaluate();
        firstLayerSigmoidDerivative.evaluate();
        firstLayerDerivativeTimesSigmoidDerivitive.evaluate();
        firstLayerDotProductWithBiasDerivative.evaluate();

        
        mergeGradWithFirstLayer.evaluate();
        mergeGradWithSecondLayer.evaluate();


    }

}

