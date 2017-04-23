cmake_minimum_required (VERSION 3.5)
project (FreeWill)

#SET (Qt5Core_DIR /usr/lib/x86_64-linux-gnu/cmake/Qt5Core)
#SET (Qt5Network_DIR /usr/lib/x86_64-linux-gnu/cmake/Qt5Network)

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

#set(CMAKE_CXX_STANDARD 14)

find_package(Qt5Core REQUIRED)
set_property(TARGET Qt5::Core PROPERTY INTERFACE_COMPILE_FEATURES "")
include_directories(${Qt5Core_INCLUDE_DIRS})
add_definitions(${Qt5Core_DEFINITIONS})
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Qt5Core_EXECUTABLE_COMPILE_FLAGS}")

find_package(Qt5Network REQUIRED)
set_property(TARGET Qt5::Network PROPERTY INTERFACE_COMPILE_FEATURES "")
include_directories(${Qt5Network_INCLUDE_DIRS})
add_definitions(${Qt5Network_DEFINITIONS})
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Qt5Network_EXECUTABLE_COMPILE_FLAGS}")


find_package(Qt5Test REQUIRED)
set_property(TARGET Qt5::Test PROPERTY INTERFACE_COMPILE_FEATURES "")
include_directories(${Qt5Test_INCLUDE_DIRS})

get_target_property(QtCore_Location Qt5::Core LOCATION)

MESSAGE("Using Qt version ${Qt5Core_VERSION} under ${QtCore_Location}")

include_directories("../../FreeWill")
include_directories("../../Utils/WebUI/Swiftly/Swiftly")
include_directories("../../Utils/WebUI/Swiftly/http-parser")
include_directories("../../Utils/WebUI/Swiftly/qt-mustache/src")
#include_directories("/usr/include/x86_64-linux-gnu/qt5/QtWebSockets")

find_package(CUDA REQUIRED)
include_directories( ${CUDA_INCLUDE_DIRS} )

MESSAGE("Using Qt version ${Qt5Core_VERSION} under ${QtCore_Location}")
set(CUDA_HOST_COMPILER /usr/bin/gcc)
list(APPEND CUDA_NVCC_FLAGS "-arch=sm_52;-std=c++11;-O3;--use_fast_math;--expt-extended-lambda")
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)


link_directories("/usr/local/cuda/lib64")

# get_directory_property(OUT_VAR LINK_DIRECTORIES)
# message(STATUS "DIR: ${OUT_VAR}")

include_directories(../../Utils/WebUI/DemoBase)

set(MNIST_SOURCES 
    MNIST.cpp
    ../../FreeWill/Tensor/Shape.cpp
    ../../FreeWill/Model/Solver.cpp
    ../../FreeWill/Model/Model.cpp
    ../../FreeWill/Model/TensorDescriptor.cpp
    ../../FreeWill/Model/OperatorDescriptor.cpp
    ../../Utils/WebUI/DemoBase/DemoBase.cpp
    ../../Utils/WebUI/DemoBase/DemoUI.cpp
    ../../Utils/WebUI/DemoBase/Session.cpp
    ../../Utils/WebUI/DemoBase/WebsocketServer.cpp
    main.cpp)

if(NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/train-images-idx3-ubyte.gz")
    MESSAGE("Downloading Dataset for MNIST ...")
    execute_process(COMMAND wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz --directory-prefix=${CMAKE_CURRENT_BINARY_DIR})
    execute_process(COMMAND wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz --directory-prefix=${CMAKE_CURRENT_BINARY_DIR})
    execute_process(COMMAND wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz --directory-prefix=${CMAKE_CURRENT_BINARY_DIR})
    execute_process(COMMAND wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz --directory-prefix=${CMAKE_CURRENT_BINARY_DIR})
    execute_process(COMMAND gzip -d ${CMAKE_CURRENT_BINARY_DIR}/t10k-images-idx3-ubyte.gz)
    execute_process(COMMAND gzip -d ${CMAKE_CURRENT_BINARY_DIR}/t10k-labels-idx1-ubyte.gz)
    execute_process(COMMAND gzip -d ${CMAKE_CURRENT_BINARY_DIR}/train-images-idx3-ubyte.gz)
    execute_process(COMMAND gzip -d ${CMAKE_CURRENT_BINARY_DIR}/train-labels-idx1-ubyte.gz)
endif()

configure_file(${CMAKE_SOURCE_DIR}/Utils/WebUI/Html/index.html ${CMAKE_CURRENT_BINARY_DIR}/Html/index.html COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/Utils/WebUI/Html/dygraph-combined-dev.js ${CMAKE_CURRENT_BINARY_DIR}/Html/dygraph-combined-dev.js COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/Utils/WebUI/Html/dat.gui.min.js ${CMAKE_CURRENT_BINARY_DIR}/Html/dat.gui.min.js COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/Utils/WebUI/Html/dygraph-extra.js ${CMAKE_CURRENT_BINARY_DIR}/Html/dygraph-extra.js COPYONLY)

add_executable(MNIST ${MNIST_SOURCES})
target_link_libraries(MNIST Qt5::Network)
target_link_libraries(MNIST Qt5::Core)
target_link_libraries(MNIST Qt5WebSockets)
target_link_libraries(MNIST Swiftly)
target_link_libraries(MNIST cuda_kernel)
target_link_libraries(MNIST ${CUDA_LIBRARIES})
target_link_libraries(MNIST cudnn)
target_link_libraries(MNIST cublas)
