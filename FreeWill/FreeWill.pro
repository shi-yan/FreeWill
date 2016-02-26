#-------------------------------------------------
#
# Project created by QtCreator 2015-05-27T20:38:43
#
#-------------------------------------------------

QT       += core gui widgets

#QT       -= gui

TARGET = FreeWill
#CONFIG   += console
CONFIG   += c++14
CONFIG   -= app_bundle

TEMPLATE = app


SOURCES += main.cpp \
    GradientCheck.cpp \
    NeuralNetwork.cpp \
    Word2VecDataset.cpp \
    StanfordSentimentDataset.cpp \
    dialog.cpp \
    NeuralNetworkThread.cpp

HEADERS += \
    GradientCheck.h \
    NeuralNetwork.h \
    ActivationFunctions.h \
    NeuralNetworkLayer.h \
    CostFunctions.h \
    Word2VecCostFunctions.h \
    Word2VecModels.h \
    Word2VecDataset.h \
    StanfordSentimentDataset.h \
    dialog.h \
    NeuralNetworkThread.h

FORMS += \
    dialog.ui
