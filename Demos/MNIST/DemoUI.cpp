#include "DemoUI.h"
#include <QFile>
#include <QFileInfo>
#include "mustache.h"
#include <QTextStream>

DemoUI::DemoUI()
{
   QFile file("index.html");
   file.open(QFile::ReadOnly);

   QTextStream ts(&file);

    QString webPageTemplate = ts.readAll();

    QVariantHash info;
    info["model_name"] = "MNIST Dataset";

    Mustache::Renderer renderer;
    Mustache::QtVariantContext context(info);

    m_content = renderer.render(webPageTemplate, &context);
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
            response << m_content;
            response.finish(HttpResponse::TEXT);
    }
}
