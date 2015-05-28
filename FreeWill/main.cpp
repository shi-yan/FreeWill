#include <QCoreApplication>
#include "GradientCheck.h"
#include "Sigmoid.h"
#include "Softmax.h"
#include <QDebug>
#include <vector>

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    std::function<double(double)> xSquare = [](double x) { return x*x;};

    double gradient = gradientCheck<double, double>(xSquare, 10, 0.0001);

    qDebug() << gradient;

    qDebug() << sigmoid<double>(3.0);

    std::vector<double> inputVector;
    inputVector.push_back(1003.0);
    inputVector.push_back(1004.0);

    std::vector<double> resultVector = softmax<double, std::vector<double>>(inputVector);

    foreach(double element, resultVector)
    {
        qDebug() << element;
    }

    return a.exec();
}
