#include <QCoreApplication>
#include "GradientCheck.h"
#include <QDebug>

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    std::function<double(double)> xSquare = [](double x) { return x*x;};

    double gradient = gradientCheck<double, double>(xSquare, 10, 0.0001);

    qDebug() << gradient;


    return a.exec();
}
