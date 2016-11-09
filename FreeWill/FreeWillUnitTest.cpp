#include "FreeWillUnitTest.h"
#include "Blob/Blob.h"

void FreeWillUnitTest::blobTest()
{
    FreeWill::Blob<FreeWill::GPU> b(32, 0, 2,3,"gg");

	QString str = "Hello";
    QVERIFY(str.toUpper() == "HELLO");
}

QTEST_MAIN(FreeWillUnitTest)
#include "FreeWillUnitTest.moc"
