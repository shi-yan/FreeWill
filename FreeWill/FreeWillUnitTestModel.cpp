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

    FreeWill::Model::TensorDescriptorHandle input = model->addTensor("input", {2}, true);
    FreeWill::Model::TensorDescriptorHandle firstLayerActivation = model->addTensor("firstLayerActivation", {2}, true);
    FreeWill::Model::TensorDescriptorHandle firstLayerDelta = model->addTensor("firstLayerDelta", {2}, true);
    FreeWill::Model::TensorDescriptorHandle secondLayerActivation = model->addTensor("secondLayerActivation", {1}, true);
    FreeWill::Model::TensorDescriptorHandle secondLayerDelta = model->addTensor("secondLayerDelta", {1}, true);
    FreeWill::Model::TensorDescriptorHandle cost = model->addTensor("cost", {1}, true);
    FreeWill::Model::TensorDescriptorHandle label = model->addTensor("label",{1}, true);

    FreeWill::Model::TensorDescriptorHandle firstLayerWeight = model->addTensor("firstLayerWeight", {2,2});
    FreeWill::Model::TensorDescriptorHandle firstLayerBias = model->addTensor("firstLayerBias", {2});
    FreeWill::Model::TensorDescriptorHandle firstLayerWeightDerivative = model->addTensor("firstLayerWeightDerivative", {2,2});
    FreeWill::Model::TensorDescriptorHandle firstLayerBiasDerivative = model->addTensor("firstLayerBiasDerivative", {2});
    FreeWill::Model::TensorDescriptorHandle secondLayerWeight = model->addTensor("secondLayerWeight",{1,2});
    FreeWill::Model::TensorDescriptorHandle secondLayerBias = model->addTensor("secondLayerBias", {1});
    FreeWill::Model::TensorDescriptorHandle secondLayerWeightDerivative = model->addTensor("secondLayerWeightDerivative", {1,2});
    FreeWill::Model::TensorDescriptorHandle secondLayerBiasDerivative = model->addTensor("secondLayerBiasDerivative", {1});
    FreeWill::Model::TensorDescriptorHandle inputNeuronDelta = model->addTensor("inputNeuronDelta", {2}, true);


    model->addOperator("firstLayerFullyConnected", FreeWill::DOT_PRODUCT_WITH_BIAS,
                        {{"Input", input}, {"Weight", firstLayerWeight}, {"Bias", firstLayerBias}},
                        {{"Output", firstLayerActivation}});
    model->addOperator("firstLayerSigmoid", FreeWill::ACTIVATION,
                        {{"Input", firstLayerActivation}}, {{"Output", firstLayerActivation}});
    model->addOperator("secondLayerFullyConnected", FreeWill::DOT_PRODUCT_WITH_BIAS,
                        {{"Input", firstLayerActivation}, {"Weight", secondLayerWeight}, {"Bias", secondLayerBias}},
                        {{"Output", secondLayerActivation}});
    model->addOperator("secondLayerSigmoid", FreeWill::ACTIVATION, {{"Input", secondLayerActivation}},{{"Output", secondLayerActivation}});
    model->addOperator("crossEntropyLoss", FreeWill::CROSS_ENTROPY_LOSS,
                        {{"Input", secondLayerActivation}, {"Label", label}},
                        {{"Cost", cost}});
    model->addOperator("sigmoidCrossEntropyLossDerivative", FreeWill::SIGMOID_CROSS_ENTROPY_LOSS_DERIVATIVE,
                        {{"Input", secondLayerActivation},{"Label", label}},{{"Output", secondLayerDelta}});
    model->addOperator("secondLayerDotProductWithBiasDerivative", FreeWill::DOT_PRODUCT_WITH_BIAS_DERIVATIVE,
                        {{"InputActivation", firstLayerActivation}, {"Weight", secondLayerWeight},{"OutputDelta", secondLayerDelta}},
                        {{"WeightGrad", secondLayerWeightDerivative}, {"BiasGrad", secondLayerBiasDerivative}, {"InputDelta", secondLayerDelta}});
    model->addOperator("firstLayerSigmoidDerivative", FreeWill::ACTIVATION_DERIVATIVE,
                        {{"Output", firstLayerActivation}, {"OutputDelta", firstLayerDelta}},{{"InputDelta", firstLayerDelta}});
    model->addOperator("firstLayerDotProductWithBiasDerivative", FreeWill::DOT_PRODUCT_WITH_BIAS_DERIVATIVE,
                        {{"InputActivation", input}, {"OutputDelta", firstLayerDelta}, {"Weight", firstLayerWeight}},
                        {{"WeightGrad", firstLayerWeightDerivative}, {"BiasGrad", firstLayerBiasDerivative},{"InputDelta", inputNeuronDelta}});

}
