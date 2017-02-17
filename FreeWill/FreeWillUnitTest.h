#include <QtTest/QtTest>

class FreeWillUnitTest : public QObject
{
	Q_OBJECT
private slots:
    void initTestCase();    
    void cleanupTestCase();
    void blobTest();
    void blobTestGPU();
    void tensorTest();
    void tensorTestGPU();
    void operatorTest();
    void operatorTestGPU();
    void operatorSigmoidTestCPUAndGPU();
    void operatorSigmoidDerivativeTest();
    void operatorSigmoidDerivativeTestGPU();
    void operatorReLUDerivativeTest();
    void operatorReLUDerivativeTestGPU();
    void operatorSigmoidCrossEntropyTestCPUAndGPU();
    void operatorSigmoidCrossEntropyDerivativeTest();
    void operatorDotProductWithBiasTest();
    void operatorDotProductWithBiasTestGPU();
    void operatorDotProductWithBiasDerivativeTest();
    void operatorDotProductWithBiasDerivativeTestGPU();
    void xorTest();
    void convNetTest();
    void convDerivativeTest();
    void SoftmaxTest();
    void SoftmaxDerivativeTest();
};
