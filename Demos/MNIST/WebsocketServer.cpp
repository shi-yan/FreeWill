#include "WebsocketServer.h"
#include <QDebug>
#include <ctime>
#include <random>
#include <endian.h>
#include <QDataStream>

WebsocketServer::WebsocketServer():
    m_testTimer(NULL)
{
    m_server = new QWebSocketServer("localhost", QWebSocketServer::NonSecureMode);

    if (m_server->listen(QHostAddress::Any, 5678))
    {
        qDebug() << "websocket listening";
        connect(m_server, &QWebSocketServer::newConnection,
                this, &WebsocketServer::onNewConnection);
        connect(m_server, &QWebSocketServer::closed, this, &WebsocketServer::onClosed);

        m_session = new Session();
        m_session->open();
    }
}

void WebsocketServer::onNewConnection()
{
    qDebug() << "new connection" ;
    QWebSocket *pSocket = m_server->nextPendingConnection();

    //connect(pSocket, &QWebSocket::textMessageReceived, this, &WebsocketServer::onTextMessageReceived);
    connect(pSocket, &QWebSocket::binaryMessageReceived, this, &WebsocketServer::onBinaryMessageReceived);
    connect(pSocket, &QWebSocket::disconnected, this, &WebsocketServer::onDisconnected);

/*    if (!m_testTimer)
    {
        m_testTimer = new QTimer(this);
        connect(m_testTimer, &QTimer::timeout, this, &WebsocketServer::onTimeout);
        m_testTimer->start(1000);
    }
*/
    m_consumerSockets.push_back(pSocket);
}

static unsigned int counter = 0;
std::mt19937 gen(std::time(NULL));
std::normal_distribution<float> normDis(0, 1);

void WebsocketServer::onTimeout()
{
    float v = (normDis(gen));
    qDebug() << "write" << v;
    m_session->write(counter++, v);
    notifyUpdate(m_session->tail());
}

void WebsocketServer::notifyUpdate(unsigned int _tail)
{
    quint32 message = UPDATE_AVAILABLE;
    quint32 tail = _tail;

    QByteArray ba;
    ba.append((char*) &message, 4);
    ba.append((char *) &tail, 4);

    foreach(QWebSocket *socket, m_consumerSockets)
    {
        socket->sendBinaryMessage(ba);
    }
}

void WebsocketServer::onBinaryMessageReceived(const QByteArray &message)
{
    quint32 messageName = 0;
    QDataStream ds(message);
    ds.setByteOrder(QDataStream::LittleEndian);
    ds >> messageName;
    qDebug() << "requesting data" << messageName << message.size();

    if (messageName == QUERY_DATA)
    {


        QWebSocket *socket = (QWebSocket*) sender();

        quint32 from = 0;
        quint32 size = 0;

        ds >> from >> size;

        qDebug() << "requesting data" << from << size;

        unsigned char *buffer = new unsigned char[(sizeof(unsigned int) + sizeof(float)) * size];

        m_session->read(buffer, from, size);

        messageName = DATA;
        QByteArray ba;
        ba.append((char*) &messageName, 4);
        ba.append((char*) &size, 4);
        ba.append((char*)buffer, (sizeof(unsigned int) + sizeof(float)) * size);

        delete [] buffer;

        socket->sendBinaryMessage(ba);
    }
}

void WebsocketServer::onClosed()
{
    qDebug() << "closed";
}

WebsocketServer::~WebsocketServer()
{

}

void WebsocketServer::onDisconnected()
{
    QWebSocket *socket = (QWebSocket*) sender();

    m_consumerSockets.removeOne(socket);
    socket->deleteLater();
}

void WebsocketServer::onUpdateCost(float cost)
{
    qDebug() << "write" << cost;
    m_session->write(counter++, cost);
    notifyUpdate(m_session->tail());
}

void WebsocketServer::onUpdateProgress(float epoch, float overall)
{
    quint32 message = UPDATE_PROGRESS;
    
    QByteArray ba;
    ba.append((char*) &message, 4);
    ba.append((char *) &epoch, 4);
    ba.append((char *) &overall, 4);

    foreach(QWebSocket *socket, m_consumerSockets)
    {
        socket->sendBinaryMessage(ba);
    }
}
