#ifndef MNIST_H
#define MNIST_H

#include <QThread>
#include <Tensor/Tensor.h>
#include "DemoBase.h"


class MNIST : public DemoBase
{
    Q_OBJECT

    FILE *datafp;
    FILE *labelfp;

    unsigned int numOfImage;
    unsigned int numOfRow;
    unsigned int numOfColumn;
    unsigned int labelCount;
   
    FILE *testDatafp;
    FILE *testLabelfp;
    
    unsigned int numOfTestImage;
    unsigned int numOfTestRow;
    unsigned int numOfTestColumn;
    unsigned int labelTestCount;

    bool m_usingConvolution;

public:
        MNIST(WebsocketServer *websocketServer, bool usingConvolution = true);
        ~MNIST();

        void openTestData();
        void openTrainData();
        void closeTestData();
        void closeTrainData();
        
        template<FreeWill::DeviceType DeviceUsed = FreeWill::DeviceType::CPU_NAIVE>
        void loadOneTrainData(FreeWill::Tensor<DeviceUsed, float> &image, FreeWill::Tensor<DeviceUsed, unsigned int> &label,unsigned int batchSize);

        template<FreeWill::DeviceType DeviceUsed = FreeWill::DeviceType::CPU_NAIVE>
        void loadOneTestData(FreeWill::Tensor<DeviceUsed, float> &image, FreeWill::Tensor<DeviceUsed, unsigned int> &label, unsigned int batchSize);


        void trainFullyConnectedModel();
        void trainConvolutionalModel();
        void trainConvolutionalModelGPU();

        void run() override;
};

#endif
