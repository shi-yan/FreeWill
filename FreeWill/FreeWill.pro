#-------------------------------------------------
#
# Project created by QtCreator 2015-05-27T20:38:43
#
#-------------------------------------------------

QT       += core gui widgets

TARGET = FreeWill
CONFIG   += c++14
CONFIG += console
CONFIG   -= app_bundle

TEMPLATE = lib

SOURCES += main.cpp \
    GradientCheck.cpp \
    NeuralNetwork.cpp \
    Word2VecDataset.cpp

HEADERS += \
    GradientCheck.h \
    NeuralNetwork.h \
    ActivationFunctions.h \
    NeuralNetworkLayer.h \
    CostFunctions.h \
    Word2VecCostFunctions.h \
    Word2VecModels.h \
    Word2VecDataset.h
