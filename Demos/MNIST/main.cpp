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
    HttpServer::getSingleton().start(QThread::idealThreadCount(), 8083);
    WebsocketServer websocketServer;

    MNIST *mnist = new MNIST(&websocketServer);
    mnist->moveToThread(mnist);
    QObject::connect(mnist, &MNIST::updateCost, &websocketServer, &WebsocketServer::onUpdateCost);
    QObject::connect(mnist, &MNIST::updateProgress, &websocketServer, &WebsocketServer::onUpdateProgress);
    mnist->start();    
    
    return a.exec();
}
