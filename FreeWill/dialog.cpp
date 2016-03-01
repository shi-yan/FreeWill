#include "dialog.h"
#include "ui_dialog.h"



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


static int nextStep(NeuralNetwork<float> &network, int testp[9])
{

    float minscore = 50;
    int minid = -1;

    std::vector<float> inputs;
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

            std::vector<float> outputs;
            network.getResult(inputs, outputs);

            float score = outputs[0];

            qDebug() <<"score:"<< i<< score;

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

Dialog::Dialog(NeuralNetwork<float> &_network, QWidget *parent) :
    network(_network),
    QDialog(parent),
    ui(new Ui::Dialog)
{
    ui->setupUi(this);

    buttons[0] = ui->pushButton_1;
    buttons[1] = ui->pushButton_2;
    buttons[2] = ui->pushButton_3;
    buttons[3] = ui->pushButton_4;
    buttons[4] = ui->pushButton_5;
    buttons[5] = ui->pushButton_6;
    buttons[6] = ui->pushButton_7;
    buttons[7] = ui->pushButton_8;
    buttons[8] = ui->pushButton_9;


    for(int i = 0; i<9;++i)
    {
        buttons[i]->setText("");
        connect(buttons[i], SIGNAL(clicked(bool)),this,SLOT(buttonClicked(bool)));
    }
}

void Dialog::buttonClicked(bool)
{
    QPushButton *senderb = (QPushButton*)sender();

    int id = -1;
    int position[9] = {0};

    for(int i =0;i<9;++i)
    {
        if( buttons[i]->text() == "")
        {
            position[i] = 0;

        }
        else if( buttons[i]->text() == "X")
        {
            position[i] = 1;
        }
        else if( buttons[i]->text() == "O")
        {
            position[i] = -1;
        }

        if (senderb == buttons[i])
        {
            id = i;
            position[i] = 1;
        }


    }



    if (id != -1)
    {
       if( buttons[id]->text() == "")
       {
           buttons[id]->setText("X");


           int first = nextStep(network, position);

           buttons[first]->setText("O");

       }
    }

    QPixmap image(size());
    render(&image);
    static int counter = 0;
    image.save(QString("image_%1.png").arg(counter++));

}

void Dialog::paintEvent(QPaintEvent *e)
{
    QDialog::paintEvent(e);


}

Dialog::~Dialog()
{
    delete ui;
}
