#include "DemoUI.h"
#include <QFile>
#include <QFileInfo>
#include "mustache.h"
#include <QTextStream>
#include <QTcpSocket>
#include "TcpSocket.h"

DemoUI::DemoUI()
{
   QFile file("./Html/index.html");
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
    QFile file(QString("./Html").append(request.getHeader().getPath()));
    if (file.exists())
    {
        if (file.open(QFile::ReadOnly))
        {
            response << file.readAll();

            QFileInfo fileInfo(file);
           // qDebug() << fileInfo.completeSuffix() << fileInfo.fileName() ;
            if (fileInfo.suffix() == "css")
            {
                response.finish(HttpResponse::TEXT, "text/css");
            }
            else if (fileInfo.suffix() == "jpg")
            {
                response.finish(HttpResponse::BINARY, "image/jpeg");
            }
            else if (fileInfo.suffix() == "js")
            {
                response.finish(HttpResponse::TEXT, "text/javascript");
            }
            else
            {
                response.finish(HttpResponse::BINARY, QString("application/").append(fileInfo.suffix()));
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
        QString address = "localhost"; //request.getLocalAddress().toString();
        info["websocket_address"] = address;//.right(address.length() - QString("::ffff:").length());

        Mustache::Renderer renderer;
        Mustache::QtVariantContext context(info);

        QString m_content = renderer.render(m_template, &context);

            response << m_content;
            response.finish(HttpResponse::TEXT);
    }
}
