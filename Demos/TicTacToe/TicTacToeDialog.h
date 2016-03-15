#ifndef TICTACTOEDIALOG_H
#define TICTACTOEDIALOG_H

#include <QDialog>
#include <QPushButton>
#include "NeuralNetwork.h"

namespace Ui {
class TicTacToeDialog;
}

class TicTacToeDialog : public QDialog
{
    Q_OBJECT


    int position[9] = {0};

    QPushButton *buttons[9] = {0};

    NeuralNetwork<double> &network;


public:
    explicit TicTacToeDialog(NeuralNetwork<double> &_network, QWidget *parent = 0);
    ~TicTacToeDialog();

private slots:
    void buttonClicked(bool);

private:
    Ui::TicTacToeDialog *ui;

    void paintEvent(QPaintEvent *) override;
};

#endif // TICTACTOEDIALOG_H
