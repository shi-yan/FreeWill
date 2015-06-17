#include "GradientCheck.h"

float xSquareFunction(const std::vector<float> &x, std::vector<float> &grad)
{
    //assume size of x is 1
    grad.resize(1);
    grad[0] = 2.0 * x[0];
    return x[0] * x[0];
}

void testGradientCheck()
{
    std::vector<float> x;
    x.push_back(5.3);

    if (gradientCheck<float>(xSquareFunction, x, 1e-4))
    {
        qDebug() << "gradient check passed!";
    }
    else
    {
        qDebug() << "gradient check failed!";
    }
}
