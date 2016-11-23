#include "FreeWillUnitTest.h"
#include "Tensor/Tensor.h"
#include "Tensor/ReferenceCountedBlob.h"

void FreeWillUnitTest::blobTest()
{
    FreeWill::ReferenceCountedBlob<FreeWill::CPU_NAIVE, float> blob1;
    blob1.alloc(10);

    QVERIFY(blob1.size() == 10);

    blob1.randomize();

    FreeWill::ReferenceCountedBlob<FreeWill::CPU_NAIVE, float> blob2;
    blob2 = blob1;

    for(unsigned int i = 0; i < blob1.size(); ++ i)
    {
        QVERIFY(blob1[i] == blob2[i]);
    }

    {
        FreeWill::ReferenceCountedBlob<FreeWill::CPU_NAIVE, float> blob3;
        blob3 = blob1.deepCopy();

        QVERIFY(blob3.size() == blob1.size());

        for(unsigned int i = 0; i < blob2.size(); ++ i)
        {
            QVERIFY(blob3[i] == blob2[i]);
        }
    }

    for(unsigned int i = 0; i < blob1.size(); ++ i)
    {
        QVERIFY(blob1[i] == blob2[i]);
    }

    //QString str = "Hello";
    //QVERIFY(str.toUpper() == "HELLO");
}

void FreeWillUnitTest::tensorTest()
{
    FreeWill::Tensor<4, FreeWill::CPU_NAIVE, float> tensor({64, 0, 32, 32});
    tensor.init();
    tensor.randomize();

    QVERIFY(1 == 1);
}

QTEST_MAIN(FreeWillUnitTest)
#include "FreeWillUnitTest.moc"
