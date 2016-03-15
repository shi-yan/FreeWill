#include <QApplication>
#include <QGuiApplication>
#include <QDebug>
#include <vector>
#include "NeuralNetwork.h"
#include "ActivationFunctions.h"
#include "CostFunctions.h"
#include "GradientCheck.h"
#include "TicTacToeDialog.h"

static double rate = 3.1;
static unsigned int count = 0;


static bool isWin(int position[9], int side)
{
if((position[0] == side && position[1] == side && position[2] == side) ||
   (position[3] == side && position[4] == side && position[5] == side) ||
   (position[6] == side && position[7] == side && position[8] == side) ||
   (position[0] == side && position[3] == side && position[6] == side)||
   (position[1] == side && position[4] == side && position[7] == side)||
   (position[2] == side && position[5] == side && position[8] == side)||
   (position[0] == side && position[4] == side && position[8] == side)||
   (position[2] == side && position[4] == side && position[6] == side))
{
    return true;
}
else
    return false;
}


float getReward(NeuralNetwork<double> &network, std::vector<double> &position, int side)
{
if((position[0] == side && position[1] == side && position[2] == side) ||
   (position[3] == side && position[4] == side && position[5] == side) ||
   (position[6] == side && position[7] == side && position[8] == side) ||
   (position[0] == side && position[3] == side && position[6] == side)||
   (position[1] == side && position[4] == side && position[7] == side)||
   (position[2] == side && position[5] == side && position[8] == side)||
   (position[0] == side && position[4] == side && position[8] == side)||
   (position[2] == side && position[4] == side && position[6] == side))
{
    if (side == 1)
    {
        return 1;
    }
    else
        return 0;
}
else
{

    std::vector<double> outputs;
    network.getResult(position, outputs);

    float q = outputs[0];
    return q;
}
}


void recursiveTrain(NeuralNetwork<double> &network, NeuralNetwork<double>::MiniBatch &miniBatch, int position[9], int side)
{

for(int i = 0;i<9;++i)
{


    if (position[i] == 0)
    {

        position[i] = side;
        float reward = 0.5;
        bool notset = true;
        bool hasWon = false;
        if (isWin(position,  side))
        {
            hasWon = true;
            if (side == 1)
            {
                reward = 1;
            }
            else
                reward = 0;
         }
        else
        {
            for(int k=0;k<9;++k)
            {
                if (position[k] !=0)
                {
                    continue;
                }
                std::vector<double> inputs;

                for(int e = 0;e<9;++e)
                {
                    inputs.push_back(position[e]);
                }
                inputs[k] = -side;

                float q = getReward(network, inputs, -side);

                if (notset)
                {
                    notset = false;
                    reward = q;
                }
                else
                if (side == 1)
                {
                    if (q < reward)
                        reward = q;
                }
                else
                {
                    if (q > reward)
                        reward = q;
                }
            }

            reward = 0.9 * reward;

        }

        //reward = (reward + 2.0 ) * 0.5

            std::vector<double> labelVector;
            labelVector.push_back(reward);


            std::vector<double> inputVector;
            inputVector.push_back(position[0]);
            inputVector.push_back(position[1]);
            inputVector.push_back(position[2]);
            inputVector.push_back(position[3]);
            inputVector.push_back(position[4]);
            inputVector.push_back(position[5]);
            inputVector.push_back(position[6]);
            inputVector.push_back(position[7]);
            inputVector.push_back(position[8]);

            NeuralNetwork<double>::TrainingData d(inputVector, labelVector);

            miniBatch.push_back(d);



        if (!hasWon)
        {
            recursiveTrain(network, miniBatch, position, -side);
        }
        position[i] = 0;
    }



}




}

static int nextStep(NeuralNetwork<double> &network, int testp[9])
{

float minscore = 50;
int minid = -1;

std::vector<double> inputs;
for(int e = 0;e<9;++e)
{
    inputs.push_back(testp[e]);
}


for(int i = 0;i<9;++i)
{
    if (testp[i] == 0)
    {
        inputs[i] = -1;
        testp[i] = -1;

        if (isWin(testp, -1))
            return i;

        std::vector<double> outputs;
        network.getResult(inputs, outputs);

        float score = outputs[0];

        if (minid == -1){
            minscore = score;
            minid = i;
        }
        else
        {
            if (score< minscore)
            {

                minscore = score;
                minid = i;
            }
        }


        inputs[i] = 0;
        testp[i] = 0;
    }

}
return minid;

}

void trainDRL(NeuralNetwork<double> &network)
{





int position[9] = {0};
int side = 1;

for (int i = 0; i< 100000; ++i)
{
    NeuralNetwork<double>::MiniBatch miniBatch;

    recursiveTrain(network,miniBatch, position, 1);

    if (miniBatch.size())
    {
        double cost;
        std::vector<NeuralNetworkLayer<double>> gradient;

        network.forwardPropagateParallel(12, miniBatch, cost, gradient);
        qDebug() << "cost" << cost << miniBatch.size();
        count++;
        if (count%1000 == 0)
        {
            rate*=0.8;
        }
        network.updateWeights(rate, gradient);
    }

    network.dumpWeights("deepreinforce_r_1_15_double", i);
    qDebug() << "save file"<< i;
}

qDebug() << "Neural network for trainDRL has trained, now test:";




/*
int a[] = {0,0,1,1};
int b[] = {0,1,0,1};
for(int i = 0; i<4; ++i)
{
    std::vector<double> inputs;
    inputs.push_back(a[i]);
    inputs.push_back(b[i]);
    std::vector<double> outputs;
    network.getResult(inputs, outputs);
    qDebug() << "a:" << a[i] << "b" << b[i] << "a XOR b" << (a[i]^b[i]) << "Neural Network computed:" << outputs[0];
}
*/



/*
int testp[9] = {0};
testp[3] = 1;

int first = nextStep(network, testp);

qDebug() << "first" << first;
testp[first] = -1;

testp[0] = 1;

int second = nextStep(network, testp);

qDebug() << "second" << second;

testp[second] = -1;

testp[5] = 1;

int third = nextStep(network, testp);

qDebug() << "third" << third;*/

}


int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    NeuralNetwork<double> network;
    std::vector<unsigned int> layerCounts;
    layerCounts.push_back(15);
    network.init(9,1,layerCounts, rectifier<double>, rectifierDerivative<double>, rectifier<double>, rectifierDerivative<double>, meanSquared<double>, derivativeMeanSquaredRectifier<double>);

    /*QFile file("deepreinforce_r_1_100_751.sav");

    file.open(QIODevice::ReadOnly);

    std::vector<double> data;
    data.resize(file.size(), 0);

    qDebug() << file.read((char*)&data[0], file.size());

    file.close();

    network.assignWeights(data);

    srand(time(NULL));
    */
     network.randomWeights();

     trainDRL(network);


    TicTacToeDialog dialog(network);

    dialog.show();
    return a.exec();
}
