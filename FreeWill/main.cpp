#include <QCoreApplication>
#include "GradientCheck.h"
#include <QDebug>
#include <vector>
#include "NeuralNetwork.h"
#include "GradientCheck.h"
#include "Word2VecModels.h"


int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    //testGradientCheck();

    testNeuralNetwork();
    return a.exec();
}
