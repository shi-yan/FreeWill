#include <QCoreApplication>
#include <QtCore/QThread>
#include "Swiftly.h"
#include "DemoUI.h"
#include "WebsocketServer.h"
#include "MNIST.h"
#include <QCommandLineParser>
#include <QMap>

int main(int argc, char *argv[])
{
    qDebug() << "MNIST Demo";
    QCoreApplication a(argc, argv);
    QCoreApplication::setApplicationName("MNIST");
    QCoreApplication::setApplicationVersion("0.1");

    QCommandLineParser parser;
    parser.setApplicationDescription("FreeWill MNIST demo");
    parser.addHelpOption();
    parser.addVersionOption();

    parser.addOptions({{{"m", "mode"}, "Select test mode [CPU_FULLYCONNECTED, CPU_FULLYCONNECTED_MODEL, CPU_CONVNET, GPU_FULLYCONNECTED, GPU_CONVNET]", "GPU_CONVNET"}});

    parser.process(a);

    QString testModeString = parser.value("mode");

    MNIST::TestMode testMode = MNIST::TestMode::GPU_CONVNET;

    if (MNIST::testModeLoopup.contains(testModeString))
    {
        testMode = MNIST::testModeLoopup[testModeString];
    }
    else
    {
        qDebug() << "Unknown test mode: " << testModeString <<". Default to GPU_CONVNET.";
    }

    REGISTER_WEBAPP(DemoUI);
    HttpServer::getSingleton().start(1, 8083);
    WebsocketServer websocketServer;

    MNIST *mnist = new MNIST(testMode, &websocketServer);
    mnist->moveToThread(mnist);
    QObject::connect(mnist, &MNIST::updateCost, &websocketServer, &WebsocketServer::onUpdateCost);
    QObject::connect(mnist, &MNIST::updateProgress, &websocketServer, &WebsocketServer::onUpdateProgress);
    mnist->start();
    
    return a.exec();
}
