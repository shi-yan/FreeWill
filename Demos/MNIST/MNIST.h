#ifndef MNIST_H
#define MNIST_H

#include <QThread>
#include <Tensor/Tensor.h>
#include "WebsocketServer.h"


class MNIST : public QThread
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

    WebsocketServer *m_websocketServer;

    bool m_usingConvolution;

public:
        MNIST(WebsocketServer *websocketServer, bool usingConvolution = true);
        ~MNIST();

        void openTestData();
        void openTrainData();
        void closeTestData();
        void closeTrainData();
        
        template<FreeWill::DeviceType DeviceUsed = FreeWill::CPU_NAIVE>
        void loadOneTrainData(FreeWill::Tensor<DeviceUsed, float> &image, FreeWill::Tensor<DeviceUsed, unsigned int> &label,unsigned int batchSize);

        template<FreeWill::DeviceType DeviceUsed = FreeWill::CPU_NAIVE>
        void loadOneTestData(FreeWill::Tensor<DeviceUsed, float> &image, FreeWill::Tensor<DeviceUsed, unsigned int> &label, unsigned int batchSize);


        void trainFullyConnectedModel();
        void trainConvolutionalModel();
        void trainConvolutionalModelGPU();

        void run();

signals:
        void updateCost(float cost);
        void updateProgress(float epoch, float overall);
};

#endif
