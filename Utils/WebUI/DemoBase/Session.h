#ifndef SESSION_H
#define SESSION_H

#include <QObject>
#include <QFile>
#include <QList>
#include <QtWebSockets/QWebSocket>

class Session : public QObject
{
    Q_OBJECT

private:

    QFile m_sessionFile;
    uchar *m_writeMap;

    unsigned int m_readFrom;
    unsigned int m_readSize;
    uchar *m_readMap;

    const unsigned int m_chunkSize = 1024;
    unsigned int m_capacity;
    unsigned int m_tail;

public:
    Session();

    void write(unsigned int step, float v);

    void open();

    void read(uchar *buffer, unsigned int offset, unsigned int size);

    unsigned int tail();
};

#endif // SESSION_H
