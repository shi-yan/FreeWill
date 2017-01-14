#ifndef WEBSOCKETSERVER_H
#define WEBSOCKETSERVER_H

#include <QtWebSockets/QWebSocket>
#include <QtWebSockets/QWebSocketServer>
#include <QTimer>
#include "Session.h"

enum MESSAGE
{
    UPDATE_AVAILABLE = 6543,
    QUERY_DATA,
    DATA
};

class WebsocketServer : public QObject
{
    Q_OBJECT

    QWebSocketServer *m_server;
    QTimer *m_testTimer;
    Session *m_session;

    QWebSocket *m_producerSocket;
    QList<QWebSocket*> m_consumerSockets;


public:
    WebsocketServer();
    virtual ~WebsocketServer();
    void notifyUpdate(unsigned int);

private slots:
    void onNewConnection();
    void onClosed();

    void onBinaryMessageReceived(const QByteArray &message);
    void onDisconnected();

    void onTimeout();

public slots:
    void onUpdateCost(float cost);

};

#endif // WEBSOCKETSERVER_H
