#include "Word2VecDialog.h"
#include "ui_Word2VecDialog.h"

Word2VecDialog::Word2VecDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::Word2VecDialog)
{
    ui->setupUi(this);
}

Word2VecDialog::~Word2VecDialog()
{
    delete ui;
}
