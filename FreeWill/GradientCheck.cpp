#include "GradientCheck.h"

double xSquareFunction(const std::vector<double> &x, std::vector<double> &grad)
{
    //assume size of x is 1
    grad.resize(1);
    grad[0] = 2.0 * x[0];
    return x[0] * x[0];
}

void testGradientCheck()
{
    std::vector<double> x;
    x.push_back(5.3);

    if (gradientCheck<double>(xSquareFunction, x, 1e-4))
    {
        qDebug() << "gradient check passed!";
    }
    else
    {
        qDebug() << "gradient check failed!";
    }
}
