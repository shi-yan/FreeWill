#ifndef DIALOG_H
#define DIALOG_H

#include <QDialog>
#include "NeuralNetwork.h"

namespace Ui {
class Dialog;
}

class Dialog : public QDialog
{
    Q_OBJECT


    int position[9] = {0};

    QPushButton *buttons[9] = {0};

    NeuralNetwork<float> &network;


public:
    explicit Dialog(NeuralNetwork<float> &_network, QWidget *parent = 0);
    ~Dialog();

private slots:
    void buttonClicked(bool);

private:
    Ui::Dialog *ui;

    void paintEvent(QPaintEvent *) override;
};

#endif // DIALOG_H
