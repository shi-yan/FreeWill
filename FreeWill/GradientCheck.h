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
    ScalarType c= func(x, gradientAtX);

    qDebug() << "gradient at x" << gradientAtX.size() << x.size();

    for(int i = 0; i< gradientAtX.size();++i)
    {
        qDebug() << gradientAtX[i];
    }

    std::vector<ScalarType> x_1 = x;
    std::vector<ScalarType> x_2 = x;

    std::vector<ScalarType> gradientAtX1;
    std::vector<ScalarType> gradientAtX2;

    for(int i = x.size()-1; i>=0; --i)
    {
        x_1[i] = x[i] - epsilon;
        x_2[i] = x[i] + epsilon;

        qDebug() << x_1[i] << ";"<< x_2[i] <<";";

        qDebug() <<  (5.2999*5.2999) << ";"<< (5.3001*5.3001);

        ScalarType valueAtX1 = func(x_1, gradientAtX1);
        ScalarType valueAtX2 = func(x_2, gradientAtX2);

        qDebug() << "cost" << c << valueAtX1 << valueAtX2;

        ScalarType numgrad = (valueAtX2 - valueAtX1) / (2.0 * epsilon);
//must use std::abs here

        qDebug() << "diff" << std::abs(numgrad - gradientAtX[i]);

        qDebug() << "max" << std::max(1.0, (double)std::max(std::abs(numgrad), std::abs(gradientAtX[i])));

        ScalarType reldiff = std::abs(numgrad - gradientAtX[i]) / std::max(1.0, (double)std::max(std::abs(numgrad), std::abs(gradientAtX[i])));
        if (reldiff > epsilon * 0.1)
        {
            qDebug() << "gradient check at" << i << "failed";
            qDebug() << "the gradient is" << gradientAtX[i] << "the numberic gradient is" << numgrad;
            qDebug() << "the error:" << reldiff;
            return false;
        }
        else
        {
            qDebug() << "gradient check at" << i << "passed";
            qDebug() << "the gradient is" << gradientAtX[i] << "the numberic gradient is" << numgrad;
            qDebug() << "the error:" << reldiff;
        }

        x_2[i] = x_1[i] = x[i];
        return true;
    }
    return true;
}

void testGradientCheck();


#endif // GRADIENTCHECK

