#if defined(__APPLE__)
#include <libkern/OSByteOrder.h>

#define htobe16(x) OSSwapHostToBigInt16(x)
#define htole16(x) OSSwapHostToLittleInt16(x)
#define be16toh(x) OSSwapBigToHostInt16(x)
#define le16toh(x) OSSwapLittleToHostInt16(x)

#define htobe32(x) OSSwapHostToBigInt32(x)
#define htole32(x) OSSwapHostToLittleInt32(x)
#define be32toh(x) OSSwapBigToHostInt32(x)
#define le32toh(x) OSSwapLittleToHostInt32(x)

#define htobe64(x) OSSwapHostToBigInt64(x)
#define htole64(x) OSSwapHostToLittleInt64(x)
#define be64toh(x) OSSwapBigToHostInt64(x)
#define le64toh(x) OSSwapLittleToHostInt64(x)
#else
#include <endian.h>
#endif

#include "Tensor/Tensor.h"
#include <cstdio>
#include <QDebug>
#include "MNIST.h"
#include <QMap>
#include <QString>

MNIST::MNIST(MNIST::TestMode testMode, WebsocketServer *websocketServer, bool usingConvolution)
    :DemoBase(websocketServer),
    datafp(NULL),
    labelfp(NULL),
    numOfImage(0),
    numOfRow(0),
    numOfColumn(0),
    labelCount(0),
    testDatafp(NULL),
    testLabelfp(NULL),
    numOfTestImage(0),
    numOfTestRow(0),
    numOfTestColumn(0),
    labelTestCount(0),
    m_usingConvolution(usingConvolution),
    m_testMode(testMode)
{
}

MNIST::~MNIST()
{
}

void MNIST::openTrainData()
{
    datafp = fopen("train-images-idx3-ubyte","rb");
    labelfp = fopen("train-labels-idx1-ubyte","rb");

    unsigned int magicNum = 0;
    numOfImage = 0;
    numOfRow = 0;
    numOfColumn = 0;

    unsigned int magicNumLabel = 0;
    labelCount = 0;

    fread(&magicNumLabel, sizeof(unsigned int), 1, labelfp);
    fread(&labelCount, sizeof(unsigned int ),1, labelfp);

    magicNumLabel = be32toh(magicNumLabel);
    labelCount = be32toh(labelCount);

    fread(&magicNum, sizeof(unsigned int), 1, datafp);
    fread(&numOfImage, sizeof(unsigned int), 1, datafp);
    fread(&numOfRow, sizeof(unsigned int), 1, datafp);
    fread(&numOfColumn, sizeof(unsigned int), 1, datafp);

    magicNum = be32toh(magicNum);
    numOfImage = be32toh(numOfImage);
    numOfRow = be32toh(numOfRow);
    numOfColumn = be32toh(numOfColumn);
}

void MNIST::closeTrainData()
{
    fclose(datafp);
    fclose(labelfp);
}

void MNIST::openTestData()
{
    testDatafp = fopen("t10k-images-idx3-ubyte","rb");
    testLabelfp = fopen("t10k-labels-idx1-ubyte","rb");

    unsigned int magicNum = 0;
    numOfTestImage = 0;
    numOfTestRow = 0;
    numOfTestColumn = 0;

    unsigned int magicNumLabel = 0;
    labelTestCount = 0;

    fread(&magicNumLabel, sizeof(unsigned int), 1, testLabelfp);
    fread(&labelTestCount, sizeof(unsigned int ),1, testLabelfp);

    magicNumLabel = be32toh(magicNumLabel);
    labelTestCount = be32toh(labelTestCount);

    fread(&magicNum, sizeof(unsigned int), 1, testDatafp);
    fread(&numOfTestImage, sizeof(unsigned int), 1, testDatafp);
    fread(&numOfTestRow, sizeof(unsigned int), 1, testDatafp);
    fread(&numOfTestColumn, sizeof(unsigned int), 1, testDatafp);

    magicNum = be32toh(magicNum);
    numOfTestImage = be32toh(numOfTestImage);
    numOfTestRow = be32toh(numOfTestRow);
    numOfTestColumn = be32toh(numOfTestColumn);
}

void MNIST::closeTestData()
{
    fclose(testDatafp);
    fclose(testLabelfp);
}

void MNIST::run()
{
    FreeWill::Context<FreeWill::DeviceType::GPU_CUDA>::getSingleton().open();

    switch(m_testMode)
    {
    case TestMode::CPU_FULLYCONNECTED:
        trainFullyConnectedModel();
        break;
    case TestMode::CPU_FULLYCONNECTED_MODEL:
        trainFullyConnectedModelWithModelClass();
        break;
    case TestMode::CPU_CONVNET:
        trainConvolutionalModel();
        break;
    case TestMode::GPU_FULLYCONNECTED:
        trainFullyConnectedModelGPU();
        break;
    case TestMode::GPU_CONVNET:
        trainConvolutionalModelGPU();
        break;
    }

    FreeWill::Context<FreeWill::DeviceType::GPU_CUDA>::getSingleton().close();
}

QMap<QString, MNIST::TestMode> MNIST::testModeLoopup = {{"CPU_FULLYCONNECTED", MNIST::TestMode::CPU_FULLYCONNECTED},
                                                        {"CPU_FULLYCONNECTED_MODEL", MNIST::TestMode::CPU_FULLYCONNECTED_MODEL},
                                                        {"CPU_CONVNET",MNIST::TestMode::CPU_CONVNET},
                                                        {"GPU_FULLYCONNECTED",MNIST::TestMode::GPU_FULLYCONNECTED},
                                                        {"GPU_CONVNET",MNIST::TestMode::GPU_CONVNET}};
