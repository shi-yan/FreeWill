//#include <QApplication>
#include <set>
#include <QByteArray>
#include <QtTest>
#include <QDebug>
#include <vector>
#include <NeuralNetwork.h>
#include <ActivationFunctions.h>
#include <CostFunctions.h>
#include <GradientCheck.h>
//#include "TicTacToeDialog.h"
#include "UnitTests/MatrixMultiplicationUnitTests.h"
#include <QElapsedTimer>

/*

static void word2Vec()
{
    StanfordSentimentDataset dataset("stanfordSentimentTreebank", 10000);

      unsigned int nToken = dataset.tokens().size();

      std::vector<std::vector<double>> inGrad;
      std::vector<std::vector<double>> outGrad;

      unsigned int previous = 303000;

      if (!previous)
      {
          inGrad.resize(nToken);
          outGrad.resize(nToken);

          for(int i = 0;i<nToken;++i)
          {
              inGrad[i].resize(10);
              outGrad[i].resize(10);

              for(int e = 0;e<10;++e)
              {
                  inGrad[i][e] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
                  outGrad[i][e] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
              }
          }

      }
      else
      {
          load(previous, inGrad, outGrad);
      }

      word2VecSGD<double>(previous, inGrad, outGrad, dataset, 1.0, 400000);

}
*/


//int main(int argc, char *argv[])
//{
/*    QApplication a(argc, argv);

    testGradientCheck();
    NeuralNetwork<float> network;
    std::vector<unsigned int> layerCounts;
    layerCounts.push_back(100);
    network.init(9,1,layerCounts, sigmoid<float>, sigmoidDerivative<float>, sigmoid<float>, sigmoidDerivative<float>, crossEntropy<float>, derivativeCrossEntropySigmoid<float>);

  QFile file("deepreinforce_r_1_100_751.sav");

    file.open(QIODevice::ReadOnly);

    std::vector<float> data;
    data.resize(file.size(), 0);

    qDebug() << file.read((char*)&data[0], file.size());

    file.close();

    network.assignWeights(data);

     srand(time(NULL));
   //  network.randomWeights();

   //  trainDRL(network);


    Dialog dialog(network);

    dialog.show();


    return a.exec();*/

   /* QApplication a(argc, argv);


    NeuralNetwork<float> network;
    std::vector<unsigned int> layerCounts;
    layerCounts.push_back(30);
    network.init(9,1,layerCounts, sigmoid<float>, sigmoidDerivative<float>, sigmoid<float>, sigmoidDerivative<float>, crossEntropy<float>, derivativeCrossEntropySigmoid<float>);
    srand(time(NULL));

    network.randomWeights();

    std::vector<float> inputs;
    inputs.resize(9, 0.5);
    std::vector<float> outputs;

    QElapsedTimer et;
    et.start();
    for(int i =0;i<10000000;++i)
    network.getResult(inputs, outputs);

    qDebug() << "done" << et.elapsed();
    return a.exec();
}
*/

QTEST_MAIN(MatrixMultiplicationUnitTests)
//#include "MatrixMultiplicationUnitTests.moc"
