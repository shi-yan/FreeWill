#ifndef COSTDIALOG_H
#define COSTDIALOG_H

#include <QDialog>

namespace Ui {
class CostDialog;
}

class CostDialog : public QDialog
{
    Q_OBJECT

    QVector<int> x;
    QVector<int> y;

    int index;


public:
    explicit CostDialog(QWidget *parent = 0);
    ~CostDialog();

private:
    void resample();

public:
    void pushData(float cost);
    void update();

private:
    Ui::CostDialog *ui;
};

#endif // COSTDIALOG_H
