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

void FreeWillUnitTest::modelXORTest()
{

    //FreeWill::RandomNumberGenerator::getSingleton().beginReplay("recordRandom.bin");
    FreeWill::Model *model = FreeWill::Model::create();

    FreeWill::TensorDescriptorHandle input = model->addTensor("input", {2}, true, false);
    FreeWill::TensorDescriptorHandle firstLayerActivation = model->addTensor("firstLayerActivation", {2}, true, false);
    FreeWill::TensorDescriptorHandle firstLayerDelta = model->addTensor("firstLayerDelta", {2}, true, false);
    FreeWill::TensorDescriptorHandle secondLayerActivation = model->addTensor("secondLayerActivation", {1}, true, false);
    FreeWill::TensorDescriptorHandle secondLayerDelta = model->addTensor("secondLayerDelta", {1}, true, false);
    FreeWill::TensorDescriptorHandle cost = model->addTensor("cost", {1}, true, false);
    FreeWill::TensorDescriptorHandle label = model->addTensor("label",{1}, true, false);

    FreeWill::TensorDescriptorHandle firstLayerWeight = model->addTensor("firstLayerWeight", {2,2});
    FreeWill::TensorDescriptorHandle firstLayerBias = model->addTensor("firstLayerBias", {2});
    FreeWill::TensorDescriptorHandle firstLayerWeightDerivative = model->addTensor("firstLayerWeightDerivative", {2,2}, false, false);
    FreeWill::TensorDescriptorHandle firstLayerBiasDerivative = model->addTensor("firstLayerBiasDerivative", {2}, false, false);
    FreeWill::TensorDescriptorHandle secondLayerWeight = model->addTensor("secondLayerWeight",{1,2});
    FreeWill::TensorDescriptorHandle secondLayerBias = model->addTensor("secondLayerBias", {1});
    FreeWill::TensorDescriptorHandle secondLayerWeightDerivative = model->addTensor("secondLayerWeightDerivative", {1,2}, false, false);
    FreeWill::TensorDescriptorHandle secondLayerBiasDerivative = model->addTensor("secondLayerBiasDerivative", {1}, false, false);
    FreeWill::TensorDescriptorHandle inputNeuronDelta = model->addTensor("inputNeuronDelta", {2}, true, false);


    FreeWill::OperatorDescriptorHandle firstLayerFullyConnected = model->addOperator("firstLayerFullyConnected", FreeWill::OperatorName::DOT_PRODUCT_WITH_BIAS,
                        {{"Input", input}, {"Weight", firstLayerWeight}, {"Bias", firstLayerBias}},
                        {{"Output", firstLayerActivation}});
    FreeWill::OperatorDescriptorHandle firstLayerSigmoid = model->addOperator("firstLayerSigmoid", FreeWill::OperatorName::ACTIVATION,
                        {{"Input", firstLayerActivation}},
                        {{"Output", firstLayerActivation}},
                        {{"Mode", FreeWill::ActivationMode::SIGMOID}});
    FreeWill::OperatorDescriptorHandle secondLayerFullyConnected = model->addOperator("secondLayerFullyConnected", FreeWill::OperatorName::DOT_PRODUCT_WITH_BIAS,
                        {{"Input", firstLayerActivation}, {"Weight", secondLayerWeight}, {"Bias", secondLayerBias}},
                        {{"Output", secondLayerActivation}});
    FreeWill::OperatorDescriptorHandle secondLayerSigmoid = model->addOperator("secondLayerSigmoid", FreeWill::OperatorName::ACTIVATION,
                        {{"Input", secondLayerActivation}},
                        {{"Output", secondLayerActivation}},
                        {{"Mode", FreeWill::ActivationMode::SIGMOID}});
    FreeWill::OperatorDescriptorHandle crossEntropyLoss = model->addOperator("crossEntropyLoss", FreeWill::OperatorName::CROSS_ENTROPY_LOSS,
                        {{"Input", secondLayerActivation}, {"Label", label}},
                        {{"Cost", cost}});
    FreeWill::OperatorDescriptorHandle sigmoidCrossEntropyLossDerivative = model->addOperator("sigmoidCrossEntropyLossDerivative", FreeWill::OperatorName::SIGMOID_CROSS_ENTROPY_LOSS_DERIVATIVE,
                        {{"Input", secondLayerActivation},{"Label", label}},
                        {{"Output", secondLayerDelta}});
    FreeWill::OperatorDescriptorHandle secondLayerDotProductWithBiasDerivative = model->addOperator("secondLayerDotProductWithBiasDerivative", FreeWill::OperatorName::DOT_PRODUCT_WITH_BIAS_DERIVATIVE,
                        {{"InputActivation", firstLayerActivation}, {"Weight", secondLayerWeight},{"OutputDelta", secondLayerDelta}},
                        {{"WeightGrad", secondLayerWeightDerivative}, {"BiasGrad", secondLayerBiasDerivative}, {"InputDelta", firstLayerDelta}});
    FreeWill::OperatorDescriptorHandle firstLayerSigmoidDerivative = model->addOperator("firstLayerSigmoidDerivative", FreeWill::OperatorName::ACTIVATION_DERIVATIVE,
                        {{"Output", firstLayerActivation}, {"OutputDelta", firstLayerDelta}},
                        {{"InputDelta", firstLayerDelta}},
                        {{"Mode",FreeWill::ActivationMode::SIGMOID}});
    FreeWill::OperatorDescriptorHandle firstLayerDotProductWithBiasDerivative = model->addOperator("firstLayerDotProductWithBiasDerivative", FreeWill::OperatorName::DOT_PRODUCT_WITH_BIAS_DERIVATIVE,
                        {{"InputActivation", input}, {"OutputDelta", firstLayerDelta}, {"Weight", firstLayerWeight}},
                        {{"WeightGrad", firstLayerWeightDerivative}, {"BiasGrad", firstLayerBiasDerivative},{"InputDelta", inputNeuronDelta}});

    model->defineForwardPath({firstLayerFullyConnected, firstLayerSigmoid, secondLayerFullyConnected, secondLayerSigmoid, crossEntropyLoss});
    model->defineBackwardPath({sigmoidCrossEntropyLossDerivative, secondLayerDotProductWithBiasDerivative, firstLayerSigmoidDerivative, firstLayerDotProductWithBiasDerivative});
    model->defineWeightUpdatePairs({{firstLayerWeight, firstLayerWeightDerivative},
                                    {firstLayerBias, firstLayerBiasDerivative},
                                    {secondLayerWeight, secondLayerWeightDerivative},
                                    {secondLayerBias, secondLayerBiasDerivative}});

    FreeWill::Solver solver;
    solver.m_deviceUsed = FreeWill::DeviceType::CPU_NAIVE;
    solver.m_batchSize = 4;
    solver.init(model);

    //FreeWill::RandomNumberGenerator::getSingleton().endReplay();

    float *inputData = model->beginMutateData(input);
    float *labelData = model->beginMutateData(label);

    for (int e = 0;e<4;++e)
    {
        int a = e & 0x1;
        int b = (e >> 1) & 0x1;
        int c = a^b;
        inputData[2*e + 0] = a;
        inputData[2*e + 1] = b;
        labelData[e+0] = c;
    }

    model->endMutateData(input);
    model->endMutateData(label);

    float learningRate = 0.02;
    for(unsigned int i = 1; i< 250000; ++i)
    {
        solver.forward(model);


        model->clearTensor(firstLayerWeightDerivative);
        model->clearTensor(secondLayerWeightDerivative);
        model->clearTensor(firstLayerBiasDerivative);
        model->clearTensor(secondLayerBiasDerivative);

        solver.backward(model);

        if (i%500000 == 0 && i!=0)
        {
           learningRate*=0.5;
        }

        solver.update(-learningRate);
    }

    std::cout << "cost:" << model->debugOutputTensor(cost) << std::endl;

    solver.forward(model);

    const float *inputDataRO = model->beginMutateData(input);
    const float *labelDataRO = model->beginMutateData(label);
    const float *resultDataRO = model->beginMutateData(secondLayerActivation);

    for (int i =0;i<4;++i)
    {
        std::cout << "test " << i << ": a " << inputDataRO[i*2] << " b " << inputDataRO[i*2+1] << " c " << labelDataRO[i] << " nn result: " << resultDataRO[i] << std::endl;
    }
}
