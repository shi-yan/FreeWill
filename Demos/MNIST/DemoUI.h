#ifndef DEMOUI_H
#define DEMOUI_H

#include "WebApp.h"

class DemoUI : public WebApp
{
    Q_OBJECT

public:
    void registerPathHandlers();
public slots:
    void handleFileGet(HttpRequest &,HttpResponse &);
};

#endif // STATICFILESERVER_H
