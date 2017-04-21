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
#include "Model/Solver.h"

void FreeWillUnitTest::modelTest()
{
    FreeWill::Model *model = FreeWill::Model::create();

    FreeWill::TensorDescriptorHandle input = model->addTensor("input", {2}, true);
    FreeWill::TensorDescriptorHandle firstLayerActivation = model->addTensor("firstLayerActivation", {2}, true);
    FreeWill::TensorDescriptorHandle firstLayerDelta = model->addTensor("firstLayerDelta", {2}, true);
    FreeWill::TensorDescriptorHandle secondLayerActivation = model->addTensor("secondLayerActivation", {1}, true);
    FreeWill::TensorDescriptorHandle secondLayerDelta = model->addTensor("secondLayerDelta", {1}, true);
    FreeWill::TensorDescriptorHandle cost = model->addTensor("cost", {1}, true);
    FreeWill::TensorDescriptorHandle label = model->addTensor("label",{1}, true);

    FreeWill::TensorDescriptorHandle firstLayerWeight = model->addTensor("firstLayerWeight", {2,2});
    FreeWill::TensorDescriptorHandle firstLayerBias = model->addTensor("firstLayerBias", {2});
    FreeWill::TensorDescriptorHandle firstLayerWeightDerivative = model->addTensor("firstLayerWeightDerivative", {2,2});
    FreeWill::TensorDescriptorHandle firstLayerBiasDerivative = model->addTensor("firstLayerBiasDerivative", {2});
    FreeWill::TensorDescriptorHandle secondLayerWeight = model->addTensor("secondLayerWeight",{1,2});
    FreeWill::TensorDescriptorHandle secondLayerBias = model->addTensor("secondLayerBias", {1});
    FreeWill::TensorDescriptorHandle secondLayerWeightDerivative = model->addTensor("secondLayerWeightDerivative", {1,2});
    FreeWill::TensorDescriptorHandle secondLayerBiasDerivative = model->addTensor("secondLayerBiasDerivative", {1});
    FreeWill::TensorDescriptorHandle inputNeuronDelta = model->addTensor("inputNeuronDelta", {2}, true);


    model->addOperator("firstLayerFullyConnected", FreeWill::DOT_PRODUCT_WITH_BIAS,
                        {{"Input", input}, {"Weight", firstLayerWeight}, {"Bias", firstLayerBias}},
                        {{"Output", firstLayerActivation}});
    model->addOperator("firstLayerSigmoid", FreeWill::ACTIVATION,
                        {{"Input", firstLayerActivation}},
                        {{"Output", firstLayerActivation}},{{"Mode", FreeWill::SIGMOID}});
    model->addOperator("secondLayerFullyConnected", FreeWill::DOT_PRODUCT_WITH_BIAS,
                        {{"Input", firstLayerActivation}, {"Weight", secondLayerWeight}, {"Bias", secondLayerBias}},
                        {{"Output", secondLayerActivation}});
    model->addOperator("secondLayerSigmoid", FreeWill::ACTIVATION,
                        {{"Input", secondLayerActivation}},
                        {{"Output", secondLayerActivation}},{{"Mode", FreeWill::SIGMOID}});
    model->addOperator("crossEntropyLoss", FreeWill::CROSS_ENTROPY_LOSS,
                        {{"Input", secondLayerActivation}, {"Label", label}},
                        {{"Cost", cost}});
    model->addOperator("sigmoidCrossEntropyLossDerivative", FreeWill::SIGMOID_CROSS_ENTROPY_LOSS_DERIVATIVE,
                                                             {{"Input", secondLayerActivation},{"Label", label}},{{"Output", secondLayerDelta}},{{"Mode", FreeWill::SIGMOID}});
    model->addOperator("secondLayerDotProductWithBiasDerivative", FreeWill::DOT_PRODUCT_WITH_BIAS_DERIVATIVE,
                        {{"InputActivation", firstLayerActivation}, {"Weight", secondLayerWeight},{"OutputDelta", secondLayerDelta}},
                        {{"WeightGrad", secondLayerWeightDerivative}, {"BiasGrad", secondLayerBiasDerivative}, {"InputDelta", firstLayerDelta}});
    model->addOperator("firstLayerSigmoidDerivative", FreeWill::ACTIVATION_DERIVATIVE,
                                                             {{"Output", firstLayerActivation}, {"OutputDelta", firstLayerDelta}},{{"InputDelta", firstLayerDelta}},{{"Mode",FreeWill::SIGMOID}});
    model->addOperator("firstLayerDotProductWithBiasDerivative", FreeWill::DOT_PRODUCT_WITH_BIAS_DERIVATIVE,
                        {{"InputActivation", input}, {"OutputDelta", firstLayerDelta}, {"Weight", firstLayerWeight}},
                        {{"WeightGrad", firstLayerWeightDerivative}, {"BiasGrad", firstLayerBiasDerivative},{"InputDelta", inputNeuronDelta}});

    FreeWill::Solver solver;
    solver.m_deviceUsed = FreeWill::CPU_NAIVE;
    solver.m_batchSize = 4;

    QVERIFY(model->init(solver));

    model->defineForwardPath({"firstLayerFullyConnected", "firstLayerSigmoid", "secondLayerFullyConnected", "secondLayerSigmoid", "crossEntropyLoss"});
    model->generateSVGDiagram("testout.svg");

}
