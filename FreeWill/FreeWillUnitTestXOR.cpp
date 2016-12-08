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
    firstLayerWeight.randomize();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> firstLayerWeightDerivative({2,3,1});
    firstLayerWeightDerivative.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> secondLayerWeight({1, 3});
    secondLayerWeight.init();
    secondLayerWeight.randomize();

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
    
    QVERIFY(firstLayerDotProductWithBiasDerivative.init()); 


    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> accumuFirstLayerWeight({2,3,1});
    accumuFirstLayerWeight.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> accumuSecondLayerWeight({1,3,1});
    accumuSecondLayerWeight.init();

    FreeWill::ElementwiseAdd<FreeWill::CPU_NAIVE, float> accumuGradForFirstLayer;
    accumuGradForFirstLayer.setInputParameter("Operand", &accumuFirstLayerWeight);
    accumuGradForFirstLayer.setInputParameter("Operand", &firstLayerWeightDerivative);
    accumuGradForFirstLayer.setOutputParameter("Result", &accumuFirstLayerWeight);
    QVERIFY(accumuGradForFirstLayer.init());

    FreeWill::ElementwiseAdd<FreeWill::CPU_NAIVE, float> accumuGradForSecondLayer;
    accumuGradForSecondLayer.setInputParameter("Operand", &accumuSecondLayerWeight);
    accumuGradForSecondLayer.setInputParameter("Operand", &secondLayerWeightDerivative);
    accumuGradForSecondLayer.setOutputParameter("Result", &accumuSecondLayerWeight);
    QVERIFY(accumuGradForSecondLayer.init());

    float learningRate = 0.2;
    FreeWill::ElementwiseAdd<FreeWill::CPU_NAIVE, float> mergeWithFirstLayer(-learningRate*0.25);
    mergeWithFirstLayer.setInputParameter("Operand", &firstLayerWeight);
    mergeWithFirstLayer.setInputParameter("Operand", &accumuFirstLayerWeight);
    mergeWithFirstLayer.setOutputParameter("Result", &firstLayerWeight);
    QVERIFY(mergeWithFirstLayer.init());

    FreeWill::ElementwiseAdd<FreeWill::CPU_NAIVE, float> mergeWithSecondLayer(-learningRate*0.25);
    mergeWithSecondLayer.setInputParameter("Operand", &secondLayerWeight);
    mergeWithSecondLayer.setInputParameter("Operand", &accumuSecondLayerWeight);
    mergeWithSecondLayer.setOutputParameter("Result", &secondLayerWeight);
    QVERIFY(mergeWithSecondLayer.init());

    float overallCost = 0.0;

    for(int i = 0; i< 10000000; ++i)
    {
        /*std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis(0, 1);
        */

       /* int a = dis(gen);
        int b = dis(gen);
        int c = a ^ b;
*/
        int a = i & 0x1;
        int b = (i >> 1) & 0x1;
        int c = a^b;
        input[0] = a;
        input[1] = b;
        label[0] = c;

        firstLayerFullyConnected.evaluate();
        firstLayerSigmoid.evaluate();

        secondLayerFullyConnected.evaluate();
        secondLayerSigmoid.evaluate();
        crossEntropy.evaluate();


 //       qDebug() << "a" << a <<"b" <<b <<"c" << c << "cost" << cost[0];

        overallCost += cost[0];
        sigmoidCrossEntropyDerivative.evaluate();
        secondLayerDotProductWithBiasDerivative.evaluate();
        firstLayerSigmoidDerivative.evaluate();
        firstLayerDerivativeTimesSigmoidDerivitive.evaluate();
        firstLayerDotProductWithBiasDerivative.evaluate();
       /* qDebug() << "-------------";
        qDebug() << "second layer sigmoid neuron" << secondLayerNeuronDerivative[0];
        
        qDebug() << "first layer sigmoid neuron" << firstLayerSigmoidNeuron[0] << firstLayerSigmoidNeuron[1] ;
        qDebug() << "first layer neuron deriv" << firstLayerNeuronDerivative[0] << firstLayerNeuronDerivative[1];
        qDebug() << "input" << input[0] << input[1];
        qDebug() << "second grad" << secondLayerWeightDerivative[0] << secondLayerWeightDerivative[1] << secondLayerWeightDerivative[2];
        qDebug() << "first grad" << firstLayerWeightDerivative[0] << firstLayerWeightDerivative[1] << firstLayerWeightDerivative[2] 
            << firstLayerWeightDerivative[3] << firstLayerWeightDerivative[4] << firstLayerWeightDerivative[5];
*/
         if (i%500000 == 0 && i!=0)
        {
           learningRate*=0.5;
        }
        accumuGradForFirstLayer.evaluate();
        accumuGradForSecondLayer.evaluate();


        if (i%4 == 0 && i!=0)
        {
          /*   qDebug() << "-------------";
        qDebug() << "second layer sigmoid neuron" << secondLayerNeuronDerivative[0];
        
        qDebug() << "first layer sigmoid neuron" << firstLayerSigmoidNeuron[0] << firstLayerSigmoidNeuron[1] ;
        qDebug() << "first layer neuron deriv" << firstLayerNeuronDerivative[0] << firstLayerNeuronDerivative[1];
        qDebug() << "input" << input[0] << input[1];
        qDebug() << "second grad" << secondLayerWeightDerivative[0] << secondLayerWeightDerivative[1] << secondLayerWeightDerivative[2];
        qDebug() << "first grad" << firstLayerWeightDerivative[0] << firstLayerWeightDerivative[1] << firstLayerWeightDerivative[2] 
            << firstLayerWeightDerivative[3] << firstLayerWeightDerivative[4] << firstLayerWeightDerivative[5];
*/

            if (i%2000 == 0)
            {
                //qDebug() <<  "cost" << overallCost*0.25;
                printf("cost: %f\n", overallCost*0.25);
            }
           overallCost = 0.0;
           mergeWithFirstLayer.setRate(-learningRate * 0.25);
           mergeWithSecondLayer.setRate(-learningRate * 0.25);
            mergeWithFirstLayer.evaluate();
            mergeWithSecondLayer.evaluate();
            accumuFirstLayerWeight.clear();
            accumuSecondLayerWeight.clear();
        }
    }

}

