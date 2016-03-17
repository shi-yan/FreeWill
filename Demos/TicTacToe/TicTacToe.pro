QT += core gui widgets printsupport

CONFIG += c++14

TARGET = TicTacToe
CONFIG -= app_bundle

TEMPLATE = app

SOURCES += main.cpp \
    TicTacToeDialog.cpp

win32:CONFIG(release, debug|release): LIBS += -L$$OUT_PWD/../../FreeWill/ -lFreeWill
else:win32:CONFIG(debug, debug|release): LIBS += -L$$OUT_PWD/../../FreeWill/ -lFreeWill
else:unix: LIBS += -L$$OUT_PWD/../../FreeWill/ -lFreeWill

INCLUDEPATH += $$PWD/../../FreeWill
DEPENDPATH += $$PWD/../../FreeWill

win32:CONFIG(release, debug|release): PRE_TARGETDEPS += $$OUT_PWD/../../FreeWill/FreeWill.lib
else:win32:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$OUT_PWD/../../FreeWill/FreeWill.lib
else:unix: PRE_TARGETDEPS += $$OUT_PWD/../../FreeWill/libFreeWill.so


win32:CONFIG(release, debug|release): LIBS += -L$$OUT_PWD/../../FreeWillUtil/ -lFreeWillUtil
else:win32:CONFIG(debug, debug|release): LIBS += -L$$OUT_PWD/../../FreeWillUtil/ -lFreeWillUtil
else:unix: LIBS += -L$$OUT_PWD/../../FreeWillUtil/ -lFreeWillUtil

INCLUDEPATH += $$PWD/../../FreeWillUtil
DEPENDPATH += $$PWD/../../FreeWillUtil

win32:CONFIG(release, debug|release): PRE_TARGETDEPS += $$OUT_PWD/../../FreeWillUtil/FreeWillUtil.lib
else:win32:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$OUT_PWD/../../FreeWillUtil/FreeWillUtil.lib
else:unix: PRE_TARGETDEPS += $$OUT_PWD/../../FreeWillUtil/libFreeWillUtil.so


HEADERS += \
    TicTacToeDialog.h

FORMS += \
    TicTacToeDialog.ui
