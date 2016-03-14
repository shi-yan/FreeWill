#-------------------------------------------------
#
# Project created by QtCreator 2016-03-14T16:04:27
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = Word2Vec
TEMPLATE = app


SOURCES += main.cpp\
        Word2VecDialog.cpp \
    StanfordSentimentDataset.cpp

HEADERS  += Word2VecDialog.h \
    StanfordSentimentDataset.h

FORMS    += Word2VecDialog.ui

win32:CONFIG(release, debug|release): LIBS += -L$$OUT_PWD/../../FreeWill/ -lFreeWill
else:win32:CONFIG(debug, debug|release): LIBS += -L$$OUT_PWD/../../FreeWill/ -lFreeWill
else:unix: LIBS += -L$$OUT_PWD/../../FreeWill/ -lFreeWill

INCLUDEPATH += $$PWD/../../FreeWill
DEPENDPATH += $$PWD/../../FreeWill

win32:CONFIG(release, debug|release): PRE_TARGETDEPS += $$OUT_PWD/../../FreeWill/FreeWill.lib
else:win32:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$OUT_PWD/../../FreeWill/FreeWill.lib
else:unix: PRE_TARGETDEPS += $$OUT_PWD/../../FreeWill/libFreeWill.so

