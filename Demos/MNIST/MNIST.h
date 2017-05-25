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
    enum class TestMode
    {
        CPU_FULLYCONNECTED,
        CPU_FULLYCONNECTED_MODEL,
        CPU_CONVNET,
        CPU_CONVNET_MODEL,
        GPU_FULLYCONNECTED,
        GPU_CONVNET
    };

    static QMap<QString, MNIST::TestMode> testModeLookup;

private:
    TestMode m_testMode;

public:
    MNIST(TestMode testMode, WebsocketServer *websocketServer, bool usingConvolution = true);
    ~MNIST();

    void openTestData();
    void openTrainData();
    void closeTestData();
    void closeTrainData();
        
    template<FreeWill::DeviceType DeviceUsed = FreeWill::DeviceType::CPU_NAIVE>
    void loadOneTrainData(FreeWill::Tensor<DeviceUsed, float> &image, FreeWill::Tensor<DeviceUsed, unsigned int> &label, unsigned int batchSize)
    {
        for(unsigned int i = 0;i<batchSize;++i)
        {
            for(unsigned int y = 0 ; y < numOfRow; ++y)
            {
                for(unsigned int x = 0;x< numOfColumn; ++x)
                {
                    unsigned char pixel = 0;
                    fread(&pixel, sizeof(unsigned char), 1, datafp);
                    image[i*numOfRow*numOfColumn + numOfColumn * y + x] = (float) pixel / 255.0f;
                }
            }

            unsigned char _label = 0;
            fread(&_label, sizeof(unsigned char), 1, labelfp);
            label[i] = _label;
        }

        if constexpr (DeviceUsed == FreeWill::DeviceType::GPU_CUDA)
        {
            image.copyFromHostToDevice();
            label.copyFromHostToDevice();
        }
    }

    void loadOneTrainData(float *image, unsigned int *label, unsigned int batchSize)
    {
        for(unsigned int i = 0;i<batchSize;++i)
        {
            for(unsigned int y = 0 ; y < numOfRow; ++y)
            {
                for(unsigned int x = 0;x< numOfColumn; ++x)
                {
                    unsigned char pixel = 0;
                    fread(&pixel, sizeof(unsigned char), 1, datafp);
                    image[i*numOfRow*numOfColumn + numOfColumn * y + x] = (float) pixel / 255.0f;
                }
            }

            unsigned char _label = 0;
            fread(&_label, sizeof(unsigned char), 1, labelfp);
            label[i] = _label;
        }
    }

    template<FreeWill::DeviceType DeviceUsed = FreeWill::DeviceType::CPU_NAIVE>
    void loadOneTestData(FreeWill::Tensor<DeviceUsed, float> &image, FreeWill::Tensor<DeviceUsed, unsigned int> &label,unsigned int batchSize)
    {
        for (unsigned int i =0;i<batchSize;++i)
        {
            for(unsigned int y = 0 ; y < numOfTestRow; ++y)
            {
                for(unsigned int x = 0;x< numOfTestColumn; ++x)
                {
                    unsigned char pixel = 0;
                    fread(&pixel, sizeof(unsigned char), 1, testDatafp);
                    image[i * numOfTestRow*numOfTestColumn +  numOfTestColumn * y + x] = (float) pixel / 255.0f;
                }
            }
            unsigned char _label = 0;
            fread(&_label, sizeof(unsigned char), 1, testLabelfp);
            label[i] = _label;
        }

        if constexpr (DeviceUsed == FreeWill::DeviceType::GPU_CUDA)
        {
            image.copyFromHostToDevice();
            label.copyFromHostToDevice();
        }
    }

    void loadOneTestData(float *image, unsigned int *label, unsigned int batchSize)
    {
        for (unsigned int i =0;i<batchSize;++i)
        {
            for(unsigned int y = 0 ; y < numOfTestRow; ++y)
            {
                for(unsigned int x = 0;x< numOfTestColumn; ++x)
                {
                    unsigned char pixel = 0;
                    fread(&pixel, sizeof(unsigned char), 1, testDatafp);
                    image[i * numOfTestRow*numOfTestColumn +  numOfTestColumn * y + x] = (float) pixel / 255.0f;
                }
            }
            unsigned char _label = 0;
            fread(&_label, sizeof(unsigned char), 1, testLabelfp);
            label[i] = _label;
        }
    }

    void trainFullyConnectedModel();
    void trainConvolutionalModel();
    void trainConvolutionalModelWithModelClass();
    void trainConvolutionalModelGPU();
    void trainFullyConnectedModelWithModelClass();
    void trainFullyConnectedModelGPU();
    void run() override;
};

#endif
