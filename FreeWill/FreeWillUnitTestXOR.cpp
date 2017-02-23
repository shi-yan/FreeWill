#include "FreeWillUnitTest.h"
#include "Tensor/Tensor.h"
#include "Operator/DotProductWithBias.h"
#include "Operator/DotProductWithBiasDerivative.h"
#include "Operator/Activation.h"
#include "Operator/ActivationDerivative.h"
#include "Operator/CrossEntropyLoss.h"
#include "Operator/SigmoidCrossEntropyLossDerivative.h"
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

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> firstLayerWeight({2,2});
    firstLayerWeight.init();
    firstLayerWeight.randomize();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> firstLayerBias({2});
    firstLayerBias.init();
    firstLayerBias.randomize();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> firstLayerWeightDerivative({2,2});
    firstLayerWeightDerivative.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> firstLayerBiasDerivative({2});
    firstLayerBiasDerivative.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> secondLayerWeight({1, 2});
    secondLayerWeight.init();
    secondLayerWeight.randomize();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> secondLayerBias({1});
    secondLayerBias.init();
    secondLayerBias.randomize();
    /*
    firstLayerWeight[0] = 0.1;
    firstLayerWeight[1] = 0.2;
    firstLayerWeight[2] = 0.3;
    firstLayerWeight[3] = 0.4;
    firstLayerWeight[4] = 0.5;
    firstLayerWeight[5] = 0.6;
    
    secondLayerWeight[0] = 0.7;
    secondLayerWeight[1] = 0.8;
    secondLayerWeight[2] = 0.9;


    printf("--------\n");
    printf("%f, %f, %f, %f, %f, %f\n",firstLayerWeight[0],
            firstLayerWeight[1], firstLayerWeight[2], firstLayerWeight[3], firstLayerWeight[4], firstLayerWeight[5]);
    printf("%f,%f,%f\n",secondLayerWeight[0], secondLayerWeight[1], secondLayerWeight[2]);
    printf("--------\n");
    */

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> secondLayerWeightDerivative({1,2});
    secondLayerWeightDerivative.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> secondLayerBiasDerivative({1});
    secondLayerBiasDerivative.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> inputNeuronDerivative({2,1});
    inputNeuronDerivative.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> firstLayerSigmoidNeuron({2,1});
    firstLayerSigmoidNeuron.init();
    
    FreeWill::DotProductWithBias<FreeWill::CPU_NAIVE, float> firstLayerFullyConnected(true);
    firstLayerFullyConnected.setInputParameter("Input", &input);
    firstLayerFullyConnected.setInputParameter("Weight", &firstLayerWeight);
    firstLayerFullyConnected.setInputParameter("Bias", &firstLayerBias);
    firstLayerFullyConnected.setOutputParameter("Output", &firstLayerActivation);
    
    QVERIFY(firstLayerActivation.init());


    FreeWill::Activation<FreeWill::SIGMOID, FreeWill::CPU_NAIVE, float> firstLayerSigmoid;
    firstLayerSigmoid.setInputParameter("Input", &firstLayerActivation);
    firstLayerSigmoid.setOutputParameter("Output", &firstLayerActivation);
    
    QVERIFY(firstLayerSigmoid.init());

    FreeWill::DotProductWithBias<FreeWill::CPU_NAIVE, float> secondLayerFullyConnected(true);
    secondLayerFullyConnected.setInputParameter("Input", &firstLayerActivation);
    secondLayerFullyConnected.setInputParameter("Weight", &secondLayerWeight);
    secondLayerFullyConnected.setInputParameter("Bias", &secondLayerBias);
    secondLayerFullyConnected.setOutputParameter("Output", &secondLayerActivation);

    QVERIFY(secondLayerFullyConnected.init());

    FreeWill::Activation<FreeWill::SIGMOID, FreeWill::CPU_NAIVE, float> secondLayerSigmoid;
    secondLayerSigmoid.setInputParameter("Input", &secondLayerActivation);
    secondLayerSigmoid.setOutputParameter("Output", &secondLayerActivation);

    QVERIFY(secondLayerSigmoid.init());

    FreeWill::CrossEntropyLoss<FreeWill::CPU_NAIVE, float> crossEntropyLoss;
    crossEntropyLoss.setInputParameter("Input", &secondLayerActivation);
    crossEntropyLoss.setInputParameter("Label", &label);
    crossEntropyLoss.setOutputParameter("Cost", &cost);

    QVERIFY(crossEntropyLoss.init());


    FreeWill::SigmoidCrossEntropyLossDerivative<FreeWill::CPU_NAIVE, float> sigmoidCrossEntropyLossDerivative;
    sigmoidCrossEntropyLossDerivative.setInputParameter("Input", &secondLayerActivation);
    sigmoidCrossEntropyLossDerivative.setInputParameter("Label", &label);
    sigmoidCrossEntropyLossDerivative.setOutputParameter("Output", &secondLayerNeuronDerivative);

    QVERIFY(sigmoidCrossEntropyLossDerivative.init());

    
    FreeWill::DotProductWithBiasDerivative<FreeWill::CPU_NAIVE, float> secondLayerDotProductWithBiasDerivative(true);
    secondLayerDotProductWithBiasDerivative.setInputParameter("InputActivation", &firstLayerActivation);
    secondLayerDotProductWithBiasDerivative.setInputParameter("OutputDelta", &secondLayerNeuronDerivative);
    secondLayerDotProductWithBiasDerivative.setInputParameter("Weight", &secondLayerWeight);

    secondLayerDotProductWithBiasDerivative.setOutputParameter("WeightGrad", &secondLayerWeightDerivative);
    secondLayerDotProductWithBiasDerivative.setOutputParameter("BiasGrad", &secondLayerBiasDerivative);
    secondLayerDotProductWithBiasDerivative.setOutputParameter("InputDelta", &firstLayerNeuronDerivative);

    QVERIFY(secondLayerDotProductWithBiasDerivative.init());

    FreeWill::ActivationDerivative<FreeWill::SIGMOID, FreeWill::CPU_NAIVE, float> firstLayerSigmoidDerivative;
    firstLayerSigmoidDerivative.setInputParameter("Output", &firstLayerActivation);
    firstLayerSigmoidDerivative.setInputParameter("OutputDelta", &firstLayerNeuronDerivative);
    firstLayerSigmoidDerivative.setOutputParameter("InputDelta", &firstLayerNeuronDerivative);

    QVERIFY(firstLayerSigmoidDerivative.init());

    
    FreeWill::DotProductWithBiasDerivative<FreeWill::CPU_NAIVE, float> firstLayerDotProductWithBiasDerivative(true);
    firstLayerDotProductWithBiasDerivative.setInputParameter("InputActivation", &input);
    firstLayerDotProductWithBiasDerivative.setInputParameter("OutputDelta", &firstLayerNeuronDerivative);
    firstLayerDotProductWithBiasDerivative.setInputParameter("Weight", &firstLayerWeight);

    firstLayerDotProductWithBiasDerivative.setOutputParameter("WeightGrad", &firstLayerWeightDerivative);
    firstLayerDotProductWithBiasDerivative.setOutputParameter("BiasGrad", &firstLayerBiasDerivative);
    firstLayerDotProductWithBiasDerivative.setOutputParameter("InputDelta", &inputNeuronDerivative);
    
    QVERIFY(firstLayerDotProductWithBiasDerivative.init()); 

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> accumuFirstLayerWeight({2,2,1});
    accumuFirstLayerWeight.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> accumuFirstLayerBias({2});
    accumuFirstLayerBias.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> accumuSecondLayerWeight({1,2,1});
    accumuSecondLayerWeight.init();

    FreeWill::Tensor<FreeWill::CPU_NAIVE, float> accumuSecondLayerBias({1});
    accumuSecondLayerBias.init();

    FreeWill::ElementwiseAdd<FreeWill::CPU_NAIVE, float> accumuGradForFirstLayerWeight;
    accumuGradForFirstLayerWeight.setInputParameter("Operand", &accumuFirstLayerWeight);
    accumuGradForFirstLayerWeight.setInputParameter("Operand", &firstLayerWeightDerivative);
    accumuGradForFirstLayerWeight.setOutputParameter("Result", &accumuFirstLayerWeight);
    QVERIFY(accumuGradForFirstLayerWeight.init());

    FreeWill::ElementwiseAdd<FreeWill::CPU_NAIVE, float> accumuGradForFirstLayerBias;
    accumuGradForFirstLayerBias.setInputParameter("Operand", &accumuFirstLayerBias);
    accumuGradForFirstLayerBias.setInputParameter("Operand", &firstLayerBiasDerivative);
    accumuGradForFirstLayerBias.setOutputParameter("Result", &accumuFirstLayerBias);
    QVERIFY(accumuGradForFirstLayerBias.init());

    FreeWill::ElementwiseAdd<FreeWill::CPU_NAIVE, float> accumuGradForSecondLayerWeight;
    accumuGradForSecondLayerWeight.setInputParameter("Operand", &accumuSecondLayerWeight);
    accumuGradForSecondLayerWeight.setInputParameter("Operand", &secondLayerWeightDerivative);
    accumuGradForSecondLayerWeight.setOutputParameter("Result", &accumuSecondLayerWeight);
    QVERIFY(accumuGradForSecondLayerWeight.init());

    FreeWill::ElementwiseAdd<FreeWill::CPU_NAIVE, float> accumuGradForSecondLayerBias;
    accumuGradForSecondLayerBias.setInputParameter("Operand", &accumuSecondLayerBias);
    accumuGradForSecondLayerBias.setInputParameter("Operand", &secondLayerBiasDerivative);
    accumuGradForSecondLayerBias.setOutputParameter("Result", &accumuSecondLayerBias);
    QVERIFY(accumuGradForSecondLayerBias.init());


    float learningRate = 0.02;
    FreeWill::ElementwiseAdd<FreeWill::CPU_NAIVE, float> mergeWithFirstLayerWeight(-learningRate*0.25);
    mergeWithFirstLayerWeight.setInputParameter("Operand", &firstLayerWeight);
    mergeWithFirstLayerWeight.setInputParameter("Operand", &accumuFirstLayerWeight);
    mergeWithFirstLayerWeight.setOutputParameter("Result", &firstLayerWeight);
    QVERIFY(mergeWithFirstLayerWeight.init());

    FreeWill::ElementwiseAdd<FreeWill::CPU_NAIVE, float> mergeWithFirstLayerBias(-learningRate*0.25);
    mergeWithFirstLayerBias.setInputParameter("Operand", &firstLayerBias);
    mergeWithFirstLayerBias.setInputParameter("Operand", &accumuFirstLayerBias);
    mergeWithFirstLayerBias.setOutputParameter("Result", &firstLayerBias);
    QVERIFY(mergeWithFirstLayerBias.init());

    FreeWill::ElementwiseAdd<FreeWill::CPU_NAIVE, float> mergeWithSecondLayerWeight(-learningRate*0.25);
    mergeWithSecondLayerWeight.setInputParameter("Operand", &secondLayerWeight);
    mergeWithSecondLayerWeight.setInputParameter("Operand", &accumuSecondLayerWeight);
    mergeWithSecondLayerWeight.setOutputParameter("Result", &secondLayerWeight);
    QVERIFY(mergeWithSecondLayerWeight.init());

    FreeWill::ElementwiseAdd<FreeWill::CPU_NAIVE, float> mergeWithSecondLayerBias(-learningRate*0.25);
    mergeWithSecondLayerBias.setInputParameter("Operand", &secondLayerBias);
    mergeWithSecondLayerBias.setInputParameter("Operand", &accumuSecondLayerBias);
    mergeWithSecondLayerBias.setOutputParameter("Result", &secondLayerBias);
    QVERIFY(mergeWithSecondLayerBias.init());

    //float overallCost = 0.0;
    
    for(unsigned int i = 1; i< 1000000; ++i)
    {
        /*std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis(0, 1);
        */

        /*int a = dis(gen);
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
        crossEntropyLoss.evaluate();
        //      qDebug() << "input" << input[0] << input[1];
        //      qDebug() << "first" << firstLayerActivation[0] << firstLayerActivation[1];
        //      qDebug() << "result" << secondLayerActivation[0] << "cost" << cost[0];
        secondLayerWeightDerivative.clear();
        secondLayerBiasDerivative.clear();

        firstLayerWeightDerivative.clear();
        firstLayerBiasDerivative.clear();
        //overallCost += cost[0];
        sigmoidCrossEntropyLossDerivative.evaluate();
        secondLayerDotProductWithBiasDerivative.evaluate();
        firstLayerSigmoidDerivative.evaluate();
        firstLayerDotProductWithBiasDerivative.evaluate();
        /* qDebug() << "-------------";
        qDebug() << "second layer sigmoid neuron" << secondLayerNeuronDerivative[0];
        
        qDebug() << "first layer sigmoid neuron" << firstLayerSigmoidNeuron[0] << firstLayerSigmoidNeuron[1] ;
        qDebug() << "first layer neuron deriv" << firstLayerNeuronDerivative[0] << firstLayerNeuronDerivative[1];
        qDebug() << "input" << input[0] << input[1];*/
        //    qDebug() << "second grad" << secondLayerWeightDerivative[0] << secondLayerWeightDerivative[1] << secondLayerWeightDerivative[2];
        //    qDebug() << "first grad" << firstLayerWeightDerivative[0] << firstLayerWeightDerivative[1] << firstLayerWeightDerivative[2] 
        //      << firstLayerWeightDerivative[3] << firstLayerWeightDerivative[4] << firstLayerWeightDerivative[5];
        
        if (i%500000 == 0 && i!=0)
        {
           learningRate*=0.5;
        }
        //  qDebug() << "second accum grad" << accumuSecondLayerWeight[0] << accumuSecondLayerWeight[1] << accumuSecondLayerWeight[2];
        //  qDebug() << "first accum grad" << accumuFirstLayerWeight[0] << accumuFirstLayerWeight[1] << accumuFirstLayerWeight[2]
        //    << accumuFirstLayerWeight[3] << accumuFirstLayerWeight[4] << accumuFirstLayerWeight[5];

        accumuGradForFirstLayerWeight.evaluate();
        accumuGradForFirstLayerBias.evaluate();
        accumuGradForSecondLayerWeight.evaluate();
        accumuGradForSecondLayerBias.evaluate();

        // qDebug() << "second accum grad" << accumuSecondLayerWeight[0] << accumuSecondLayerWeight[1] << accumuSecondLayerWeight[2];
        //      qDebug() << "first accum grad" << accumuFirstLayerWeight[0] << accumuFirstLayerWeight[1] << accumuFirstLayerWeight[2]
        //        << accumuFirstLayerWeight[3] << accumuFirstLayerWeight[4] << accumuFirstLayerWeight[5];

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
       
            //if (i%2000 == 0)
            //{
                //qDebug() <<  "cost" << overallCost*0.25;
                //printf("cost: %f\n", overallCost);
                //qDebug() << "============================";
            //}
        
            //overallCost = 0.0;
            mergeWithFirstLayerWeight.setRate(-learningRate * 0.25);
            mergeWithFirstLayerBias.setRate(-learningRate * 0.25);
            mergeWithSecondLayerWeight.setRate(-learningRate * 0.25);
            mergeWithSecondLayerBias.setRate(-learningRate * 0.25);
            mergeWithFirstLayerWeight.evaluate();
            mergeWithFirstLayerBias.evaluate();
            mergeWithSecondLayerWeight.evaluate();
            mergeWithSecondLayerBias.evaluate();

            //printf("updated weights: %f, %f, %f, %f,%f, %f\n", firstLayerWeight[0], firstLayerWeight[1], firstLayerWeight[2], firstLayerWeight[3], firstLayerWeight[4],
            //firstLayerWeight[5]);
            //printf("update weights: %f, %f, %f\n", secondLayerWeight[0], secondLayerWeight[1],secondLayerWeight[2]);
            accumuFirstLayerWeight.clear();
            accumuFirstLayerBias.clear();
            accumuSecondLayerWeight.clear();
            accumuSecondLayerBias.clear();
        }
        
    }

    qDebug() << "cost:" << cost[0];

    for (int i = 0; i< 4 ;++i)
    {
    
        int a = i & 0x1;
        int b = (i >> 1) & 0x1;

        int c = a ^ b ;

        input[0] = a;
        input[1] = b;
        label[0] = c;

        firstLayerFullyConnected.evaluate();

        firstLayerSigmoid.evaluate();

        secondLayerFullyConnected.evaluate();
        secondLayerSigmoid.evaluate();
 
        qDebug() << "test" << i << ": a" << a << "b" << b << "c" << c << "nn result:" << secondLayerActivation[0];
    }
}

