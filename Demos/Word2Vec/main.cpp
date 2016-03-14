#include "Word2VecDialog.h"
#include <QApplication>
#include "Word2VecModels.h"
#include "StanfordSentimentDataset.h"

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


int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    Word2VecDialog w;
    w.show();

    return a.exec();
}
