#ifndef DEMOUI_H
#define DEMOUI_H

#include "WebApp.h"

class DemoUI : public WebApp
{
    Q_OBJECT

        //QString m_content;
        QString m_template;
public:
        DemoUI();
        ~DemoUI();
        void registerPathHandlers();
public slots:
        void handleFileGet(HttpRequest &,HttpResponse &);
};

#endif // STATICFILESERVER_H
