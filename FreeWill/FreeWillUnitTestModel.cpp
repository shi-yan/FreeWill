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

void FreeWillUnitTest::modelTest()
{
    FreeWill::Model *model = FreeWill::Model::create();

    model->addTensor("input", {2}, true);
    model->addTensor("firstLayerActivation", {2}, true);
    model->addTensor("firstLayerDelta", {2}, true);
    model->addTensor("secondLayerActivation", {1}, true);
    model->addTensor("secondLayerDelta", {1}, true);
    model->addTensor("cost", {1}, true);
    model->addTensor("label",{1}, true);

    model->addTensor("firstLayerWeight", {2,2});
    model->addTensor("firstLayerBias", {2});
    model->addTensor("firstLayerWeightDerivative", {2,2});
    model->addTensor("firstLayerBiasDerivative", {2});
    model->addTensor("secondLayerWeight",{1,2});
    model->addTensor("secondLayerBias", {1});
    model->addTensor("secondLayerWeightDerivative", {1,2});
    model->addTensor("secondLayerBiasDerivative", {1});
    model->addTensor("inputNeuronDelta", {2}, true);


    model->addOperator("firstLayerFullyConnected", FreeWill::DOT_PRODUCT_WITH_BIAS, {{"Input", "input"},
                       {"Weight", "firstLayerWeight"},
                       {"Bias", "firstLayerBias"},
                       {"Output", "firstLayerActivation"}});

    model->addOperator("firstLayerSigmoid", FreeWill::ACTIVATION, {});
    model->addOperator("secondLayerFullyConnected", FreeWill::DOT_PRODUCT_WITH_BIAS, {});
    model->addOperator("secondLayerSigmoid", FreeWill::ACTIVATION, {});
    model->addOperator("crossEntropyLoss", FreeWill::CROSS_ENTROPY_LOSS, {});
    model->addOperator("sigmoidCrossEntropyLossDerivative", FreeWill::SIGMOID_CROSS_ENTROPY_LOSS_DERIVATIVE, {});
    model->addOperator("secondLayerDotProductWithBiasDerivative", FreeWill::DOT_PRODUCT_WITH_BIAS_DERIVATIVE, {});
    model->addOperator("firstLayerSigmoidDerivative", FreeWill::ACTIVATION_DERIVATIVE, {});
    model->addOperator("firstLayerDotProductWithBiasDerivative", FreeWill::DOT_PRODUCT_WITH_BIAS_DERIVATIVE, {});


 /*   FreeWill::Activation<FreeWill::SIGMOID, FreeWill::CPU_NAIVE, float> firstLayerSigmoid;
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

*/

}
