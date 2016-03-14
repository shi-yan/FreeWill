#include "Word2VecDialog.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    Word2VecDialog w;
    w.show();

    return a.exec();
}
