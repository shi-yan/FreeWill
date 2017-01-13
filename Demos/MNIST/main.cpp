#include <QCoreApplication>
#include <QtCore/QThread>
#include "Swiftly.h"
#include "DemoUI.h"
#include "WebsocketServer.h"
#include "MNIST.h"

int main(int argc, char *argv[])
{
    qDebug() << "MNIST Demo";
    QCoreApplication a(argc, argv);
    REGISTER_WEBAPP(DemoUI);
    HttpServer::getSingleton().start(QThread::idealThreadCount(), 80);
    WebsocketServer websocketServer;
    //websocketServer.listen(QHostAddress::Any, 5678);
    return a.exec();
}
