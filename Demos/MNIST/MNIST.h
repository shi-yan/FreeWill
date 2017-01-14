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
        void loadOneTrainData(FreeWill::Tensor<FreeWill::CPU, float> &image, FreeWill::Tensor<FreeWill::CPU, unsigned int> &label);
        void loadOneTestData(FreeWill::Tensor<FreeWill::CPU, float> &image, FreeWill::Tensor<FreeWill::CPU, unsigned int> &label);


        void trainFullyConnectedModel();
        void trainConvolutionalModel();

        void run();

signals:
        void updateCost(float cost);
};

#endif
