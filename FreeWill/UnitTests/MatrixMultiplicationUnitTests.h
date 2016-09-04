#ifndef MATRIXMULTIPLICATIONUNITTESTS_H
#define MATRIXMULTIPLICATIONUNITTESTS_H

#include <QtTest/QtTest>

class MatrixMultiplicationUnitTests:public QObject
{
    Q_OBJECT
protected slots:
    void initTestCase();
private slots:
    void testSmallMatrix();
    void testLargeMatrix();
    void testSpecialMatrix_32_x();
    void testSpecialMatrix_x_32();
    void testSpecialMatrix_32_32();
    void testCostFunctionCPUvsGPUCrossEntropySigmoid();
    void testCostFunctionCPUvsGPUMeanSquaredRectifier();
    void gradientCheckOnCrossEntropySigmoidCost();
    void gradientCheckOnMeanSquaredRectifierCost();

};

#endif // MATRIXMULTIPLICATIONUNITTESTS_H
