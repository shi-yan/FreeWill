#ifndef MNIST_H
#define MNIST_H

#include <QObject>
#include <Tensor/Tensor.h>

class MNIST : public QObject
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

public:
        MNIST();
        ~MNIST();

        void openTestData();
        void openTrainData();
        void closeTestData();
        void closeTrainData();
        void loadOneTrainData(FreeWill::Tensor<FreeWill::CPU, float> &image, FreeWill::Tensor<FreeWill::CPU, unsigned int> &label);
        void loadOneTestData(FreeWill::Tensor<FreeWill::CPU, float> &image, FreeWill::Tensor<FreeWill::CPU, unsigned int> &label);


        void trainFullyConnectedModel();
        void trainConvolutionalModel();
};

#endif
