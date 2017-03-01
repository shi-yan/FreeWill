#include "DemoUI.h"
#include <QFile>
#include <QFileInfo>
#include "mustache.h"
#include <QTextStream>
#include <QTcpSocket>
#include "TcpSocket.h"
#include <QHostAddress>

DemoUI::DemoUI()
{
   QFile file("index.html");
   file.open(QFile::ReadOnly);

   QTextStream ts(&file);

    m_template = ts.readAll();

    file.close();
}

DemoUI::~DemoUI(){}

void DemoUI::registerPathHandlers()
{
   addGetHandler("/", "handleFileGet");
}

void DemoUI::handleFileGet(HttpRequest &request, HttpResponse &response)
{
    //qDebug() << "-----------%%%" << request.getHeader().getPath();

    //qDebug() << request.getHeader().getHeaderInfo();
    QFile file(QString(".").append(request.getHeader().getPath()));
    if (file.exists())
    {
        if (file.open(QFile::ReadOnly))
        {
            response << file.readAll();

            QFileInfo fileInfo(file);
           // qDebug() << fileInfo.completeSuffix() << fileInfo.fileName() ;
           // if (fileInfo.completeSuffix() == "html" || fileInfo.completeSuffix() == "htm")
            {
                response.finish(HttpResponse::TEXT);
            }
            //else
            {
              //  response.finish(HttpResponse::BINARY);
            }
            file.close();
        }
    }
    else
    {
//        response.setStatusCode(404);
//        response << "can't find the file!\n";

        QVariantHash info;
        info["model_name"] = "MNIST Dataset";
        QString address = request.getLocalAddress().toString();
        info["websocket_address"] = address.right(address.length() - QString("::ffff:").length());

        Mustache::Renderer renderer;
        Mustache::QtVariantContext context(info);

        QString m_content = renderer.render(m_template, &context);

            response << m_content;
            response.finish(HttpResponse::TEXT);
    }
}
