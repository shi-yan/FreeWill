#include "FreeWillUnitTest.h"
#include "Tensor/Tensor.h"

void FreeWillUnitTest::blobTest()
{
	FreeWill::Tensor<1, FreeWill::GPU> a((FreeWill::Shape<1>()));
    FreeWill::Tensor<2, FreeWill::CPU> b((FreeWill::Shape<2>()));

	QString str = "Hello";
    QVERIFY(str.toUpper() == "HELLO");
}

QTEST_MAIN(FreeWillUnitTest)
#include "FreeWillUnitTest.moc"
