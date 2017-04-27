#include "FreeWillUnitTest.h"
#include "Tensor/Tensor.h"
#include "Tensor/ReferenceCountedBlob.h"
#include "Operator/Operator.h"
#include "Operator/ElementwiseAdd.h"
#include <time.h>
#include <cuda_runtime.h>
#include "Context/Context.h"

void FreeWillUnitTest::initTestCase()
{
    srand(/*time(NULL)*/0);

    //FreeWill::Context<FreeWill::DeviceType::GPU_CUDA>::getSingleton().open();
}

void FreeWillUnitTest::cleanupTestCase()
{
    //FreeWill::Context<FreeWill::DeviceType::GPU_CUDA>::getSingleton().close();
}

void FreeWillUnitTest::blobTest()
{
    FreeWill::ReferenceCountedBlob<FreeWill::DeviceType::CPU_NAIVE> blob1;
    blob1.alloc(10);

    QVERIFY(blob1.sizeInByte() == 10);

    FreeWill::ReferenceCountedBlob<FreeWill::DeviceType::CPU_NAIVE> blob2;
    blob2 = blob1;

    for(unsigned int i = 0; i < blob1.sizeInByte(); ++ i)
    {
        QVERIFY(blob1[i] == blob2[i]);
    }

    {
        FreeWill::ReferenceCountedBlob<FreeWill::DeviceType::CPU_NAIVE> blob3;
        blob3 = blob1.deepCopy();

        QVERIFY(blob3.sizeInByte() == blob1.sizeInByte());

        for(unsigned int i = 0; i < blob2.sizeInByte(); ++ i)
        {
            QVERIFY(blob3[i] == blob2[i]);
        }
    }

    for(unsigned int i = 0; i < blob1.sizeInByte(); ++ i)
    {
        QVERIFY(blob1[i] == blob2[i]);
    }

    //QString str = "Hello";
    //QVERIFY(str.toUpper() == "HELLO");
}

void FreeWillUnitTest::blobTestGPU()
{
    FreeWill::ReferenceCountedBlob<FreeWill::DeviceType::GPU_CUDA> blob1;
    blob1.alloc(10);

    QVERIFY(blob1.sizeInByte() == 10);

    FreeWill::ReferenceCountedBlob<FreeWill::DeviceType::GPU_CUDA> blob2;

    blob2 = blob1;

    blob1.copyFromDeviceToHost();
    blob2.copyFromDeviceToHost();

    for (unsigned int i = 0; i < blob1.sizeInByte(); ++i)
    {
        QVERIFY(blob1[i] == blob2[i]);
    }

    FreeWill::ReferenceCountedBlob<FreeWill::DeviceType::GPU_CUDA> blob3;
    blob3 = blob1.deepCopy();

    QVERIFY(blob3.sizeInByte() == blob1.sizeInByte());

    blob3.copyFromDeviceToHost();
    blob2.copyFromDeviceToHost();
    for(unsigned int i = 0;i<blob2.sizeInByte(); ++i)
    {
        QVERIFY(blob3[i] == blob2[i]);
    }

    blob1.copyFromDeviceToHost();
    blob2.copyFromDeviceToHost();
    for(unsigned int i = 0;i<blob1.sizeInByte();++i)
    {
        QVERIFY(blob1[i] == blob2[i]);
    }

}

void FreeWillUnitTest::tensorTest()
{
    FreeWill::Tensor< FreeWill::DeviceType::CPU_NAIVE, float> tensor({64, 32, 32});
    tensor.init();
    
    unsigned int tensorSize = tensor.shape().size();

    for(unsigned int i = 0;i<tensorSize;++i)
    {
        QVERIFY(tensor[i] == 0);
    }

    tensor.randomize();

    tensor.clear();

    for(unsigned int i = 0;i<tensorSize;++i)
    {
        QVERIFY(tensor[i] == 0);
    }

    auto tensor2 = new FreeWill::Tensor< FreeWill::DeviceType::CPU_NAIVE, float>({10});

    delete tensor2;

    //QVERIFY(1 == 1);
}

void FreeWillUnitTest::tensorTestGPU()
{
    FreeWill::Tensor<FreeWill::DeviceType::GPU_CUDA, float> tensor({64,32,32});
    tensor.init();

    unsigned int tensorSize = tensor.shape().size();

    tensor.copyFromDeviceToHost();
    for(unsigned int i = 0;i<tensorSize;++i)
    {
        QVERIFY(tensor[i] == 0);
    }

    tensor.randomize();
    tensor.copyFromHostToDevice();
    tensor.clear();
    tensor.copyFromHostToDevice();
    tensor.copyFromDeviceToHost();

    for(unsigned int i = 0;i<tensorSize;++i)
    {
        QVERIFY(tensor[i] == 0);
    }

    auto tensor2 = new FreeWill::Tensor<FreeWill::DeviceType::GPU_CUDA, float>({10});

    delete tensor2;

}

void FreeWillUnitTest::operatorTest()
{
//    FreeWill::Operator<FreeWill::CPU> o;

    FreeWill::Tensor< FreeWill::DeviceType::CPU_NAIVE, float> tensorA({64,  32, 32});
    tensorA.init();
    tensorA.randomize();

    FreeWill::Tensor< FreeWill::DeviceType::CPU_NAIVE, float> tensorB({64,  32, 32});
    tensorB.init();
    tensorB.randomize();

    FreeWill::Tensor< FreeWill::DeviceType::CPU_NAIVE, float> result({64,  32, 32});
    result.init();
    

    FreeWill::ElementwiseAdd< FreeWill::DeviceType::CPU_NAIVE, float> elementAdd;
    elementAdd.setInputParameter("OperandA", &tensorA);
    elementAdd.setInputParameter("OperandB", &tensorB);
    elementAdd.setOutputParameter("Result", &result);

    elementAdd.init();
    elementAdd.evaluate();


    unsigned int size = tensorA.shape().size();

    for(unsigned int i = 0; i<size; ++i)
    {
        QVERIFY(result[i] == (tensorA[i] + tensorB[i]));
    }
}

void FreeWillUnitTest::operatorTestGPU()
{
    FreeWill::Tensor<FreeWill::DeviceType::GPU_CUDA, float> tensorA({64,32,32});
    tensorA.init();
    tensorA.randomize();

    FreeWill::Tensor<FreeWill::DeviceType::GPU_CUDA, float> tensorB({64,32,32});
    tensorB.init();
    tensorB.randomize();

    FreeWill::Tensor<FreeWill::DeviceType::GPU_CUDA, float> result({64,32,32});
    result.init();

    FreeWill::ElementwiseAdd<FreeWill::DeviceType::GPU_CUDA, float> elementAdd;
    elementAdd.setInputParameter("OperandA", &tensorA);
    elementAdd.setInputParameter("OperandB", &tensorB);
    elementAdd.setOutputParameter("Result", &result);

    unsigned int size = tensorA.shape().size();

    elementAdd.init();

    tensorA.copyFromHostToDevice();
    tensorB.copyFromHostToDevice();

    elementAdd.evaluate();

    result.copyFromDeviceToHost();

    for(unsigned int i = 0;i< size;++i)
    {
        QVERIFY(result[i] == (tensorA[i] + tensorB[i]));
    }
}
