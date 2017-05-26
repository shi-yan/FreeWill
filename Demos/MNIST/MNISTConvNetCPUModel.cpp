#include <cstdio>
#include <QDebug>
#include "MNIST.h"
#include "Model/Model.h"
#include "Model/Solver.h"
#include "Context/Context.h"

#include <chrono>

void MNIST::trainConvolutionalModelWithModelClass()
{
    qDebug() << "================== CPU Fully Connected network with model class ==========================";

    FreeWill::Context<FreeWill::DeviceType::CPU_NAIVE>::getSingleton().open();

    FreeWill::Model *model = FreeWill::Model::create();

    const unsigned int featureMapSize = 20;
    const unsigned int batchSize = 2;

    //batch random
    FreeWill::TensorDescriptorHandle image = model->addTensor("image", {1,28,28}).enableBatch();

    FreeWill::TensorDescriptorHandle label = model->addTensor("label", {1}, FreeWill::DataType::UNSIGNED_INT).enableBatch();

    FreeWill::TensorDescriptorHandle featureMap = model->addTensor("featureMap", {1,5,5, featureMapSize}).randomize();

    FreeWill::TensorDescriptorHandle bias = model->addTensor("bias", {featureMapSize}).randomize();

    FreeWill::TensorDescriptorHandle convOutput = model->addTensor("convOutput", {featureMapSize, 24,24}).enableBatch();

    FreeWill::TensorDescriptorHandle poolingOutput = model->addTensor("poolingOutput", {featureMapSize, 12,12}).enableBatch();

    FreeWill::TensorDescriptorHandle poolingSwitchX = model->addTensor("poolingSwitchX",{featureMapSize, 12,12}, FreeWill::DataType::UNSIGNED_INT).enableBatch();

    FreeWill::TensorDescriptorHandle poolingSwitchY = model->addTensor("poolingSwitchY", {featureMapSize, 12,12}, FreeWill::DataType::UNSIGNED_INT).enableBatch();

    FreeWill::TensorDescriptorHandle fullyConnected1Weight = model->addTensor("fullyConnected1Weight", {100, featureMapSize*12*12}).randomize();

    FreeWill::TensorDescriptorHandle fullyConnected1Bias = model->addTensor("fullyConnected1Bias", {100}).randomize();

    FreeWill::TensorDescriptorHandle fullyConnected1Output = model->addTensor("fullyConnected1Output", {100}).enableBatch();

    FreeWill::TensorDescriptorHandle fullyConnected2Weight = model->addTensor("fullyConnected2Weight", {10,100}).randomize();

    FreeWill::TensorDescriptorHandle fullyConnected2Bias = model->addTensor("fullyConnected2Bias", {10}).randomize();

    FreeWill::TensorDescriptorHandle fullyConnected2Output = model->addTensor("fillyConnected2Output", {10}).enableBatch();

    FreeWill::TensorDescriptorHandle softmaxOutput = model->addTensor("softmaxOutput", {10}).enableBatch();

    FreeWill::TensorDescriptorHandle cost = model->addTensor("cost", {1}).enableBatch();

    FreeWill::TensorDescriptorHandle softmaxGrad = model->addTensor("softmaxGrad", {10}).enableBatch();

    FreeWill::TensorDescriptorHandle fullyConnected1OutputGrad = model->addTensor("fullyConnected1OutputGrad", {100}).enableBatch();

    FreeWill::TensorDescriptorHandle fullyConnected2WeightGrad = model->addTensor("fullyConnected2WeightGrad", {10,100});

    FreeWill::TensorDescriptorHandle fullyConnected2BiasGrad = model->addTensor("fullyConnected2BiasGrad", {10});

    FreeWill::TensorDescriptorHandle poolingOutputGrad = model->addTensor("poolingOutputGrad", {featureMapSize*12*12}).enableBatch();

    FreeWill::TensorDescriptorHandle fullyConnected1WeightGrad = model->addTensor("fullyConnect1WeightGrad", {100, featureMapSize*12*12});

    FreeWill::TensorDescriptorHandle fullyConnected1BiasGrad = model->addTensor("fullyConnected1BiasGrad", {100});

    FreeWill::TensorDescriptorHandle convOutputGrad = model->addTensor("convOutputGrad", {featureMapSize,24,24}).enableBatch();

    FreeWill::TensorDescriptorHandle convBiasGrad = model->addTensor("convBiasGrad", {featureMapSize});

    FreeWill::TensorDescriptorHandle convFeatureMapGrad = model->addTensor("convFeatureMapGrad", {1,5,5,featureMapSize});

    FreeWill::TensorDescriptorHandle inputGrad = model->addTensor("inputGrad", {1,28,28}).enableBatch();


    FreeWill::OperatorDescriptorHandle convolution = model->addOperator("convolution", FreeWill::OperatorName::CONVOLUTION,
    {{"Input", image}, {"FeatureMap", featureMap}, {"Bias", bias}},{{"Output", convOutput}});

    FreeWill::OperatorDescriptorHandle convSigmoid = model->addOperator("convSigmoid", FreeWill::OperatorName::ACTIVATION,
    {{"Input", convOutput}},{{"Output", convOutput}},{{"Mode", FreeWill::ActivationMode::SIGMOID}});

    /*FreeWill::OperatorDescriptorHandle reshapeBeforeMaxPooling = model->addOperator("reshapeBeforeMaxPooling", FreeWill::OperatorName::RESHAPE,
    {{"Tensor", poolingOutput}}, {}, {{"NewShape", Shape({featureMapSize*12*12})}});*/

    FreeWill::OperatorDescriptorHandle maxPooling = model->addOperator("maxPooling", FreeWill::OperatorName::MAX_POOLING,
    {{"Input", convOutput}}, {{"Output", poolingOutput.reshape({featureMapSize, 12, 12})}, {"SwitchX", poolingSwitchX},{"SwitchY", poolingSwitchY}});

    /*FreeWill::OperatorDescriptorHandle reshapeAfterMaxPooling = model->addOperator("reshapeAfterMaxPooling", FreeWill::OperatorName::RESHAPE,
    {{"Tensor", poolingOutput}}, {}, {{"NewShape", Shape({featureMapSize, 12,12})}});*/

    FreeWill::OperatorDescriptorHandle fullyConnected1 = model->addOperator("fullyConnected1", FreeWill::OperatorName::DOT_PRODUCT_WITH_BIAS,
    {{"Input", poolingOutput.reshape({featureMapSize* 12* 12})}, {"Weight", fullyConnected1Weight}, {"Bias", fullyConnected1Bias}}, {{"Output", fullyConnected1Output}});

    FreeWill::OperatorDescriptorHandle sigmoid1 = model->addOperator("sigmoid1", FreeWill::OperatorName::ACTIVATION,
    {{"Input", fullyConnected1Output}}, {{"Output", fullyConnected1Output}},
    {{"Mode", FreeWill::ActivationMode::SIGMOID}});

    FreeWill::OperatorDescriptorHandle fullyConnected2 = model->addOperator("fullyConnected2", FreeWill::OperatorName::DOT_PRODUCT_WITH_BIAS,
    {{"Input", fullyConnected1Output}, {"Weight", fullyConnected2Weight}, {"Bias", fullyConnected2Bias}},{{"Output", fullyConnected2Output}});

    FreeWill::OperatorDescriptorHandle softmaxLogLoss = model->addOperator("softmaxLogLoss", FreeWill::OperatorName::SOFTMAX_LOG_LOSS,
    {{"Input", fullyConnected2Output}, {"Label", label}}, {{"Output", softmaxOutput},{"Cost", cost}});

    FreeWill::OperatorDescriptorHandle softmaxLogLossDerivative = model->addOperator("softmaxLogLossDerivative", FreeWill::OperatorName::SOFTMAX_LOG_LOSS_DERIVATIVE,
    {{"Output", softmaxOutput},{"Label", label}},{{"InputGrad", softmaxGrad}});

    FreeWill::OperatorDescriptorHandle dotProductWithBias2Derivative = model->addOperator("dotProductWithBias2Derivative", FreeWill::OperatorName::DOT_PRODUCT_WITH_BIAS_DERIVATIVE,
    {{"InputActivation", fullyConnected1Output},{"OutputDelta", softmaxGrad},{"Weight", fullyConnected2Weight}},
    {{"InputDelta", fullyConnected1OutputGrad}, {"BiasGrad", fullyConnected2BiasGrad}, {"WeightGrad", fullyConnected2WeightGrad}},
    {{"Mode", FreeWill::ActivationMode::SIGMOID}});

    FreeWill::OperatorDescriptorHandle sigmoidDerivative = model->addOperator("sigmoidDerivative", FreeWill::OperatorName::ACTIVATION_DERIVATIVE,
    {{"Output", fullyConnected1Output}, {"OutputDelta", fullyConnected1OutputGrad}}, {{"InputDelta", fullyConnected1OutputGrad}},
    {{"Mode", FreeWill::ActivationMode::SIGMOID}});

    FreeWill::OperatorDescriptorHandle dotProductWithBias1Derivative = model->addOperator("dotProductWithBias1Derivative", FreeWill::OperatorName::DOT_PRODUCT_WITH_BIAS_DERIVATIVE,
    {{"InputActivation", poolingOutput.reshape({featureMapSize*12*12})},{"OutputDelta", fullyConnected1OutputGrad},{"Weight", fullyConnected1Weight}},
    {{"InputDelta", poolingOutputGrad.reshape({featureMapSize*12*12})},{"BiasGrad", fullyConnected1BiasGrad},{"WeightGrad", fullyConnected1WeightGrad}});

    FreeWill::OperatorDescriptorHandle maxPoolingDerivative = model->addOperator("maxPoolingDerivative", FreeWill::OperatorName::MAX_POOLING_DERIVATIVE,
    {{"OutputGrad", poolingOutputGrad.reshape({featureMapSize, 12, 12})}, {"SwitchX", poolingSwitchX}, {"SwitchY", poolingSwitchY}},{{"InputGrad", convOutputGrad}});

    FreeWill::OperatorDescriptorHandle convSigmoidDerivative = model->addOperator("convSigmoidDerivative", FreeWill::OperatorName::ACTIVATION_DERIVATIVE,
    {{"Output", convOutput},{"OutputDelta", convOutputGrad}}, {{"InputDelta", convOutputGrad}},
    {{"Mode", FreeWill::ActivationMode::SIGMOID}});

    FreeWill::OperatorDescriptorHandle convDerivative = model->addOperator("convDerivative", FreeWill::OperatorName::CONVOLUTION_DERIVATIVE,
    {{"PrevActivation", image}, {"FeatureMap", featureMap}, {"OutputGrad", convOutputGrad}},
    {{"FeatureMapGrad", convFeatureMapGrad}, {"BiasGrad", convBiasGrad}, {"InputGrad", inputGrad}});


    model->defineForwardPath({convolution, convSigmoid, maxPooling, fullyConnected1, sigmoid1, fullyConnected2, softmaxLogLoss});
    model->defineBackwardPath({softmaxLogLossDerivative, dotProductWithBias2Derivative, sigmoidDerivative, dotProductWithBias1Derivative,
                               maxPoolingDerivative, convSigmoidDerivative, convDerivative});
    model->defineWeightUpdatePairs({{fullyConnected2Weight, fullyConnected2WeightGrad},
                                    {fullyConnected2Bias, fullyConnected2BiasGrad},
                                    {fullyConnected1Weight, fullyConnected1WeightGrad},
                                    {fullyConnected1Bias, fullyConnected1BiasGrad},
                                    {featureMap, convFeatureMapGrad},
                                    {bias, convBiasGrad}});

    FreeWill::Solver solver;
    solver.m_deviceUsed = FreeWill::DeviceType::CPU_NAIVE;
    solver.m_batchSize = batchSize;
    if (!solver.init(model))
    {
        delete model;
        FreeWill::Context<FreeWill::DeviceType::CPU_NAIVE>::getSingleton().close();
        return;
    }

    float learningRate = 0.05;

    float overallCost = 0.0;
    const int testInterval = 2000;

    const int deviceCount = FreeWill::Context<FreeWill::DeviceType::CPU_NAIVE>::getSingleton().deviceCount();

    for(unsigned int e = 1; e<=60; ++e)
    {
        openTrainData();
        auto startTime = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::nano> forwardTime = std::chrono::duration<double, std::nano>::zero();
        std::chrono::duration<double, std::nano> backwardTime = std::chrono::duration<double, std::nano>::zero();
        for(unsigned int i = 1; i<=numOfImage/(batchSize*deviceCount); ++i)
        {
            for(int d = 0; d<deviceCount;++d)
            {
                float *inputData = model->beginMutateData(image, d);
                unsigned int *labelData = model->beginMutateData<FreeWill::DeviceType::CPU_NAIVE, unsigned int>(label,d);

                loadOneTrainData(inputData, labelData, batchSize);

                model->endMutateData(image,d);
                model->endMutateData(label,d);
            }
            auto forwardStartTime = std::chrono::steady_clock::now();
            solver.forward(model);
            auto forwardEndTime = std::chrono::steady_clock::now();
            forwardTime += std::chrono::duration <double, std::nano>(forwardEndTime - forwardStartTime);

            for(int d = 0; d<deviceCount;++d)
            {
                const float *costData = model->readonlyAccess(cost);

                for(unsigned int c = 0; c<batchSize; ++c)
                {
                    overallCost += costData[c];
                }
            }

            qDebug() << e << i<< "cost" << overallCost / (float) (batchSize*deviceCount) << (learningRate / (float)(deviceCount));

            emit updateCost(overallCost / (float) (batchSize*deviceCount));
            overallCost = 0.0;

            model->clearTensor(softmaxGrad );
            model->clearTensor(fullyConnected1OutputGrad );
            model->clearTensor(fullyConnected2WeightGrad );
            model->clearTensor(fullyConnected1WeightGrad );
            model->clearTensor(fullyConnected1BiasGrad );
            model->clearTensor(fullyConnected2BiasGrad );
            model->clearTensor(poolingOutputGrad );
            model->clearTensor(convOutputGrad );
            model->clearTensor(convBiasGrad );
            model->clearTensor(convFeatureMapGrad );
            model->clearTensor(inputGrad );

            auto backwardStartTime = std::chrono::steady_clock::now();

            solver.backward(model);

            auto backwardEndTime = std::chrono::steady_clock::now();
            backwardTime += std::chrono::duration <double, std::nano>(backwardEndTime - backwardStartTime);


            solver.update(-learningRate/(float)(deviceCount*batchSize));

            if (i%30000 == 0)
            {
               learningRate *= 0.9;
            }


            emit updateProgress(i*(batchSize*deviceCount) / (float)(numOfImage), ((e-1)*numOfImage + i*batchSize*deviceCount) / (60.0f*numOfImage));

        }
        auto endTime = std::chrono::steady_clock::now();
        auto diff = endTime - startTime;

        qDebug() << std::chrono::duration <double, std::milli> (diff).count() << " ms used for 1 epoch";
        qDebug() << std::chrono::duration <double, std::milli> (forwardTime).count() << " ms used for forward";
        qDebug() << std::chrono::duration <double, std::milli> (backwardTime).count() << " ms used for backward";

        closeTrainData();

        if (/*i % testInterval == 0*/true)
        {
            unsigned int correct = 0;

            openTestData();

            for (unsigned int v = 0;v<numOfTestImage/(batchSize*deviceCount); ++v)
            {
                model->clearTensor(convOutput);
                model->clearTensor(poolingOutput);
                model->clearTensor(poolingSwitchX);
                model->clearTensor(poolingSwitchY);
                model->clearTensor(fullyConnected1Output);
                model->clearTensor(fullyConnected2Output);
                model->clearTensor(softmaxOutput);
                model->clearTensor(cost);
                model->clearTensor(softmaxGrad);

                for(unsigned int d = 0;d<deviceCount;++d)
                {
                    float *inputData = model->beginMutateData(image,d);
                    unsigned int *labelData = model->beginMutateData<FreeWill::DeviceType::CPU_NAIVE, unsigned int>(label,d);


                    loadOneTestData(inputData, labelData, batchSize);

                    model->endMutateData(image,d);
                    model->endMutateData(label,d);
                }

                solver.forward(model);

                for(unsigned int d = 0;d<deviceCount;++d)
                {

                    for (unsigned int b = 0; b<batchSize; ++b)
                    {
                        unsigned int maxIndex = 0;

                        const float *softmaxOutputData = model->readonlyAccess(softmaxOutput,d);

                        float maxValue = softmaxOutputData[b * 10];

                        for(unsigned int e = 1; e < 10; ++e)
                        {
                            if (maxValue < softmaxOutputData[b* 10 + e])
                            {
                                maxValue = softmaxOutputData[b* 10 + e];
                                maxIndex = e;
                            }
                        }

                        const unsigned int *labelData = model->readonlyAccess<FreeWill::DeviceType::CPU_NAIVE, unsigned int>(label,d);

                        if ((float) maxIndex == labelData[b])
                        {
                            correct ++;
                        }
                    }
                }


            }
            qDebug() << "Accuracy" << (float) correct / (float) numOfTestImage;

            closeTestData();


            model->clearTensor(convOutput );
            model->clearTensor(poolingOutput );
            model->clearTensor(poolingSwitchX );
            model->clearTensor(poolingSwitchY );
            model->clearTensor(fullyConnected1Output );
            model->clearTensor(fullyConnected2Output );
            model->clearTensor(softmaxOutput );
            model->clearTensor(cost );
            model->clearTensor(softmaxGrad );
            model->clearTensor(fullyConnected1OutputGrad );
            model->clearTensor(fullyConnected2WeightGrad );
            model->clearTensor(fullyConnected1WeightGrad );
            model->clearTensor(fullyConnected1BiasGrad );
            model->clearTensor(fullyConnected2BiasGrad );
            model->clearTensor(poolingOutputGrad );
            model->clearTensor(convOutputGrad );
            model->clearTensor(convBiasGrad );
            model->clearTensor(convFeatureMapGrad );
            model->clearTensor(inputGrad );

        }


        //break;

    }

    delete model;


    FreeWill::Context<FreeWill::DeviceType::CPU_NAIVE>::getSingleton().close();

}
