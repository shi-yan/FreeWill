cmake_minimum_required (VERSION 3.5)
project (FreeWill)

#SET (Qt5Core_DIR /usr/lib/x86_64-linux-gnu/cmake/Qt5Core)
#SET (Qt5Test_DIR /usr/lib/x86_64-linux-gnu/cmake/Qt5Test)
#SET (CMAKE_PREFIX_PATH /usr/lib/x86_64-linux-gnu/qt5)
#SET (CMAKE_C_COMPILER /home/shiy/gcc7/bin/gcc)
SET (CMAKE_C_FLAGS          "-std=gnu++1z -march=x86-64 -m64 -Wno-c++1z-extensions")
#SET (CMAKE_CXX_COMPILER /home/shiy/gcc7/bin/g++)
SET (CMAKE_CXX_FLAGS        "-std=gnu++1z -march=x86-64 -m64 -fno-omit-frame-pointer -fPIC -I/usr/local/include -Wall -Wextra -Woverloaded-virtual -Wno-unused-local-typedefs")

set (CMAKE_C_FLAGS          "${CMAKE_C_FLAGS}" CACHE STRING "c flags")
set (CMAKE_CXX_FLAGS        "${CMAKE_CXX_FLAGS}" CACHE STRING "c++ flags")
SET (CMAKE_EXPORT_COMPILE_COMMANDS 1)
set (CMAKE_EXPORT_COMPILE_COMMANDS "${CMAKE_EXPORT_COMPILE_COMMANDS}" CACHE STRING "export compile_commands.json")

SET(GCC_COVERAGE_COMPILE_FLAGS "-fprofile-arcs -ftest-coverage")
SET(GCC_COVERAGE_LINK_FLAGS    "-lgcov")

option(DEBUG_PREPROCESSOR "Debug preprocessor" OFF)
SET(GCC_DEBUG_PRECOMPILER_FLAGS "-E")

if(DEBUG_PREPROCESSOR)
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${GCC_DEBUG_PRECOMPILER_FLAGS}")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_DEBUG_PRECOMPILER_FLAGS}")
endif(DEBUG_PREPROCESSOR)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release CACHE STRING
      "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)
set (CMAKE_BUILD_TYPE "${CMAKE_BUILD_TYPE}" CACHE STRING "build type")

SET (CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
# Instruct CMake to run moc automatically when needed.
set(CMAKE_AUTOMOC ON)

#set(CMAKE_CXX_STANDARD 17)

find_package(Qt5Core REQUIRED)
set_property(TARGET Qt5::Core PROPERTY INTERFACE_COMPILE_FEATURES "")
include_directories(${Qt5Core_INCLUDE_DIRS})
add_definitions(${Qt5Core_DEFINITIONS})
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Qt5Core_EXECUTABLE_COMPILE_FLAGS}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pthread")

find_package(Qt5Test REQUIRED)
set_property(TARGET Qt5::Test PROPERTY INTERFACE_COMPILE_FEATURES "")
include_directories(${Qt5Test_INCLUDE_DIRS})

get_target_property(QtCore_Location Qt5::Core LOCATION)


find_package(CUDA REQUIRED)
include_directories( ${CUDA_INCLUDE_DIRS} )

MESSAGE("Using Qt version ${Qt5Core_VERSION} under ${QtCore_Location}")
set(CUDA_HOST_COMPILER /usr/bin/gcc)
list(APPEND CUDA_NVCC_FLAGS "-arch=sm_52;-std=c++11;-O3;--use_fast_math;--expt-extended-lambda")
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

cuda_add_library(cuda_kernel 
                 Operator/ElementwiseAdd_CUDA.cu 
                 Operator/ElementwiseAdd_CUDA.h
                 Operator/CrossEntropyLoss_CUDA.cu
                 Operator/CrossEntropyLoss_CUDA.h
                 Operator/SoftmaxLogLoss_CUDA.h
                 Operator/SoftmaxLogLoss_CUDA.cu)

if("${CMAKE_SYSTEM_NAME}" STREQUAL "Darwin")

link_directories("/usr/local/cuda/lib")

else()

link_directories("/usr/local/cuda/lib64")

endif()

# get_directory_property(OUT_VAR LINK_DIRECTORIES)
# message(STATUS "DIR: ${OUT_VAR}")

set(FreeWill_SOURCES
    FreeWillUnitTestTensor.cpp
    FreeWillUnitTest.cpp
    FreeWillUnitTestXOR.cpp
    FreeWillUnitTestConvNet.cpp
    FreeWillUnitTestActivation.cpp
    FreeWillUnitTestModel.cpp
    Tensor/Tensor.h
    Tensor/ReferenceCountedBlob.h
    Tensor/Shape.h
    Operator/Activation.h
    Operator/ActivationDerivative.h
    Operator/DotProductWithBias.h
    Operator/SoftmaxLogLoss.h
    Operator/ElementwiseAdd.h
    Operator/ElementwiseProduct.h
    Operator/Operator.h
    Operator/CrossEntropyLoss.h
    Operator/SigmoidCrossEntropyLossDerivative.h
    Operator/SoftmaxLogLossDerivative.h
    Operator/Convolution.h
    DeviceSelection.h
    Operator/ConvolutionDerivative.h
    Operator/DotProductWithBiasDerivative.h
    Operator/MaxPooling.h
    Operator/MaxPoolingDerivative.h
    Context/Context.h
    Context/Device.h
    Context/Device.cpp
    Context/DeviceCPU.cpp
    Model/Model.h
    Model/Model.cpp
    Model/TensorDescriptor.h
    Model/TensorDescriptor.cpp
    Model/OperatorDescriptor.h
    Model/OperatorDescriptor.cpp
    Model/Solver.h
    Tensor/Shape.cpp
    Model/Solver.cpp
    Tensor/RandomNumberGenerator.h
    Tensor/RandomNumberGenerator.cpp
    )

add_executable(FreeWillUnitTest ${FreeWill_SOURCES})
target_link_libraries(FreeWillUnitTest Qt5::Core)
target_link_libraries(FreeWillUnitTest Qt5::Test)
target_link_libraries(FreeWillUnitTest cuda_kernel)
target_link_libraries(FreeWillUnitTest ${CUDA_LIBRARIES})
target_link_libraries(FreeWillUnitTest cudnn)
target_link_libraries(FreeWillUnitTest cublas)
