#ifndef WORD2VECDIALOG_H
#define WORD2VECDIALOG_H

#include <QDialog>

namespace Ui {
class Word2VecDialog;
}

class Word2VecDialog : public QDialog
{
    Q_OBJECT

public:
    explicit Word2VecDialog(QWidget *parent = 0);
    ~Word2VecDialog();

private:
    Ui::Word2VecDialog *ui;
};

#endif // WORD2VECDIALOG_H
