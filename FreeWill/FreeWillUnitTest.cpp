#include "FreeWillUnitTest.h"

void FreeWillUnitTest::firstTest()
{
	QString str = "Hello";
    QVERIFY(str.toUpper() == "HELLO");
}

QTEST_MAIN(FreeWillUnitTest)
#include "FreeWillUnitTest.moc"