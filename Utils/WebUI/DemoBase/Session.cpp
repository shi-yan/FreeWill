#include "Session.h"
#include <QDebug>

Session::Session()
    :m_sessionFile("sessionfile.dat"),
      m_writeMap(NULL),
      m_readMap(NULL),
      m_readFrom(0),
      m_readSize(0),
      m_capacity(0),
      m_tail(0)
{

}

void Session::open()
{
    m_capacity = 1024;
    m_tail = 0;

    if (!m_sessionFile.open(QFile::ReadWrite))
    {
        qDebug() << "can't open";
        return;
    }

    if (m_sessionFile.size() == 0)
    {
        //extend file
        m_sessionFile.seek(1024*1024*32);
        char buf[256] = {1};
        m_sessionFile.write(buf, 256);
        m_sessionFile.seek(0);
    }

    m_writeMap = m_sessionFile.map(m_tail * (sizeof(unsigned int) + sizeof(float)), m_chunkSize * (sizeof(unsigned int) + sizeof(float)));

    if (!m_writeMap)
    {
        qDebug() << "bug! can't open file";
    }

}

void Session::write(unsigned int step, float v)
{
    if (m_tail == m_capacity)
    {
        m_capacity += m_chunkSize;


        m_sessionFile.unmap(m_writeMap);
        m_writeMap = 0;

        m_writeMap = m_sessionFile.map(/*m_tail * (sizeof(unsigned int) + sizeof(float))*/ 0, m_capacity * (sizeof(unsigned int) + sizeof(float)));

        if (!m_writeMap)
        {

            qDebug() << "bug! can't open file";
        }
    }

    memcpy(m_writeMap + (sizeof(unsigned int) + sizeof(float)) * m_tail, &step, sizeof(unsigned int));
    memcpy(m_writeMap + (sizeof(unsigned int) + sizeof(float)) * m_tail + sizeof(unsigned int), &v, sizeof(float));
    m_tail ++;
}

unsigned int Session::tail()
{
    return m_tail;
}

void Session::read(uchar *buffer, unsigned int offset, unsigned int size)
{
    /*if (m_readMap == NULL || (offset < m_readFrom || (offset + size) > (m_readFrom + m_readSize)))
    {
        if (m_readMap)
        {
            m_sessionFile.unmap(m_readMap);
        }

        m_readMap = m_sessionFile.map(offset * (sizeof(unsigned int) + sizeof(float)), size * (sizeof(unsigned int) + sizeof(float)));
        m_readFrom = offset;
        m_readSize = size;

    }*/

    memcpy(buffer, m_writeMap + offset * (sizeof(unsigned int) + sizeof(float)), size* (sizeof(unsigned int) + sizeof(float)));


}
