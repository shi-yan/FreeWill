#include "FreeWillUnitTest.h"
#include "Tensor/Tensor.h"
#include "Tensor/ReferenceCountedBlob.h"
#include "Model.h"
#include "Operator/Operator.h"
#include "Operator/ElementwiseAdd.h"
#include "Operator/Sigmoid.h"

void FreeWillUnitTest::blobTest()
{
    FreeWill::ReferenceCountedBlob<FreeWill::CPU_NAIVE> blob1;
    blob1.alloc(10);

    QVERIFY(blob1.sizeInByte() == 10);

    FreeWill::ReferenceCountedBlob<FreeWill::CPU_NAIVE> blob2;
    blob2 = blob1;

    for(unsigned int i = 0; i < blob1.sizeInByte(); ++ i)
    {
        QVERIFY(blob1[i] == blob2[i]);
    }

    {
        FreeWill::ReferenceCountedBlob<FreeWill::CPU_NAIVE> blob3;
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

void FreeWillUnitTest::tensorTest()
{
    FreeWill::Tensor< FreeWill::CPU_NAIVE, float> tensor({64, 0, 32, 32});
    tensor.init();
    tensor.randomize();

    auto tensor2 = new FreeWill::Tensor< FreeWill::CPU_NAIVE, float>({10});

    delete tensor2;



    QVERIFY(1 == 1);
}

void FreeWillUnitTest::operatorTest()
{
//    FreeWill::Operator<FreeWill::CPU> o;

    FreeWill::Tensor< FreeWill::CPU_NAIVE, float> tensorA({64, 0, 32, 32});
    tensorA.init();
    tensorA.randomize();

    FreeWill::Tensor< FreeWill::CPU_NAIVE, float> tensorB({64, 0, 32, 32});
    tensorB.init();
    tensorB.randomize();

    FreeWill::Tensor< FreeWill::CPU_NAIVE, float> result({64, 0, 32, 32});
    result.init();
    

    FreeWill::ElementwiseAdd< FreeWill::CPU_NAIVE, float> elementAdd;
    elementAdd.setInputParameter("Operand", &tensorA);
    elementAdd.setInputParameter("Operand", &tensorB);
    elementAdd.setOutputParameter("Result", &result);

    elementAdd.init();
    elementAdd.evaluate();


    unsigned int size = tensorA.shape().size();

    for(unsigned int i = 0; i<size; ++i)
    {
        QVERIFY(result[i] == (tensorA[i] + tensorB[i]));
    }
}

void FreeWillUnitTest::operatorSigmoidTest()
{
    FreeWill::Tensor< FreeWill::CPU_NAIVE, float> input({64,0,32,32});
    input.init();
    input.randomize();

    FreeWill::Tensor< FreeWill::CPU_NAIVE, float> output({64,0,32,32});
    output.init();

    FreeWill::Sigmoid< FreeWill::CPU_NAIVE, float> sigmoid;
    sigmoid.setInputParameter("Input", &input);
    sigmoid.setOutputParameter("Output", &output);

    sigmoid.init();
    sigmoid.evaluate();
}

QTEST_MAIN(FreeWillUnitTest)
#include "FreeWillUnitTest.moc"
