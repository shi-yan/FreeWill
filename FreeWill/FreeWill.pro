#-------------------------------------------------
#
# Project created by QtCreator 2015-05-27T20:38:43
#
#-------------------------------------------------

QT       += core testlib

TARGET = FreeWill
CONFIG   += c++14
CONFIG += console
CONFIG   -= app_bundle

TEMPLATE = app

SOURCES += \
    GradientCheck.cpp \
    NeuralNetwork.cpp \
    Word2VecDataset.cpp \
    UnitTests/MatrixMultiplicationUnitTest.cpp

HEADERS += \
    GradientCheck.h \
    NeuralNetwork.h \
    ActivationFunctions.h \
    CostFunctions.h \
    Word2VecCostFunctions.h \
    Word2VecModels.h \
    Word2VecDataset.h \
    FullyConnectedLayer.h \
    FreeWill.h \
    FullyConnectedLayerKernelGPU.h \
    Global.h \
    UnitTests/MatrixMultiplicationUnitTests.h \
    CostFunctionsGPU.h


CUDA_OBJECTS_DIR = release/cuda

# This makes the .cu files appear in your project
OTHER_FILES += 

# CUDA settings <-- may change depending on your system
CUDA_SOURCES += FullyConnectedLayerKernelGPU.cu \
                CostFunctionsGPU.cu
CUDA_SDK = "/usr/local/cuda-8.0/bin/nvcc"   # Path to cuda SDK install
CUDA_DIR = "/usr/local/cuda-8.0"            # Path to cuda toolkit install
SYSTEM_NAME = x64         # Depending on your system either 'Win32', 'x64', or 'Win64'
SYSTEM_TYPE = 64            # '32' or '64', depending on your system
CUDA_ARCH = sm_30           # Type of CUDA architecture, for example 'compute_10', 'compute_11', 'sm_10'
NVCC_OPTIONS = --use_fast_math

# include paths
INCLUDEPATH += $$CUDA_DIR/include
#               $$CUDA_SDK/common/inc/ \
#               $$CUDA_SDK/../shared/inc/

# library directories
QMAKE_LIBDIR += $$CUDA_DIR/lib64
#                $$CUDA_SDK/common/lib/$$SYSTEM_NAME \
#                $$CUDA_SDK/../shared/lib/$$SYSTEM_NAME
# Add the necessary libraries
LIBS += -lcuda -lcudart

# The following library conflicts with something in Cuda
#QMAKE_LFLAGS_RELEASE = /NODEFAULTLIB:msvcrt.lib
#QMAKE_LFLAGS_DEBUG   = /NODEFAULTLIB:msvcrtd.lib

# The following makes sure all path names (which often include spaces) are put between quotation marks
CUDA_INC = $$join(INCLUDEPATH,'" -I"','-I"','"')

# Configuration of the Cuda compiler
CONFIG(debug, debug|release) {
    # Debug mode
    cuda_d.input = CUDA_SOURCES
    cuda_d.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
    cuda_d.commands = $$CUDA_DIR/bin/nvcc -D_DEBUG $$NVCC_OPTIONS $$CUDA_INC $$LIBS --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME} -Xcompiler -fPIC
    cuda_d.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda_d
}
else {
    # Release mode
    cuda.input = CUDA_SOURCES
    cuda.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
    cuda.commands = $$CUDA_DIR/bin/nvcc $$NVCC_OPTIONS $$CUDA_INC $$LIBS --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME} -Xcompiler -fPIC
    cuda.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda
}

DISTFILES += \
    ActivationFunctions.cuh \
    FullyConnectedLayerKernelGPU.cu \
    CostFunctionsGPU.cu




