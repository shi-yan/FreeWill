#ifndef DEMOBASE_H
#define DEMOBASE_H

#include <QObject>
#include <QThread>
#include "WebsocketServer.h"

class DemoBase : public QThread
{
    Q_OBJECT

protected:
    WebsocketServer *m_websocketServer;


    DemoBase(WebsocketServer *websocketServer);

    ~DemoBase();

signals:
        void updateCost(float cost);
        void updateProgress(float epoch, float overall);
};

#endif
