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
    void operatorSigmoidCrossEntropyDerivativeTestGPU();
    void operatorDotProductWithBiasTest();
    void operatorDotProductWithBiasTestGPU();
    void operatorDotProductWithBiasDerivativeTest();
    void operatorDotProductWithBiasDerivativeTestGPU();
    void SoftmaxTest();
    void SoftmaxTestGPU();
    void SoftmaxDerivativeTest();
    void SoftmaxDerivativeTestGPU();
    void convolutionTest();
    void convolutionTestGPU();
    void convolutionDerivativeTest();
    void xorTest();
};
