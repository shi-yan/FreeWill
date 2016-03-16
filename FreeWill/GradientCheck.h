#ifndef GRADIENTCHECK
#define GRADIENTCHECK

#include <functional>
#include <vector>
#include <QDebug>
#include <algorithm>
#include <cmath>

template<class ScalarType>
bool gradientCheck(std::function<ScalarType(const std::vector<ScalarType> &, std::vector<ScalarType> &)> func, std::vector<ScalarType> x, ScalarType epsilon)
{
    std::vector<ScalarType> gradientAtX;
    func(x, gradientAtX);

    std::vector<ScalarType> x_1 = x;
    std::vector<ScalarType> x_2 = x;

    std::vector<ScalarType> gradientAtX1;
    std::vector<ScalarType> gradientAtX2;

    for(int i = x.size() - 1; i >= 0; --i)
    {

        x_1[i] = x[i] - epsilon;
        x_2[i] = x[i] + epsilon;

        ScalarType valueAtX1 = func(x_1, gradientAtX1);
        ScalarType valueAtX2 = func(x_2, gradientAtX2);
        ScalarType numgrad = (valueAtX2 - valueAtX1) / (2.0 * epsilon);
        //must use std::abs here

        ScalarType reldiff = std::abs(numgrad - gradientAtX[i]) / std::max(1.0, (double)std::max(std::abs(numgrad), std::abs(gradientAtX[i])));
        if (reldiff > epsilon * 0.1)
        {
            qDebug() << "gradient check at" << i << "failed" << x[i];
            qDebug() << "the gradient is" << gradientAtX[i] << "the numberic gradient is" << numgrad;
            qDebug() << "the error:" << reldiff;
            return false;
        }
        else
        {
            /*qDebug() << "gradient check at" << i << "passed";
            qDebug() << "the gradient is" << gradientAtX[i] << "the numberic gradient is" << numgrad;
            qDebug() << "the error:" << reldiff;*/
        }

        x_2[i] = x_1[i] = x[i];
    }
    return true;
}

void testGradientCheck();


#endif // GRADIENTCHECK

