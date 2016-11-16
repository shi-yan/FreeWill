cmake_minimum_required (VERSION 3.5)
project (FreeWill)

SET (CMAKE_C_COMPILER /Users/shiyan/clang/build/bin/clang)
SET (CMAKE_C_FLAGS          "-std=gnu++1z -march=x86-64 -m64")
SET (CMAKE_CXX_COMPILER /usr/bin/g++)
SET (CMAKE_CXX_FLAGS        "-std=c++1z -march=x86-64 -m64 -fno-omit-frame-pointer -fPIC -I/usr/local/include -Wall -Wextra -Woverloaded-virtual -Wno-unused-local-typedefs")

set (CMAKE_C_FLAGS          "${CMAKE_C_FLAGS}" CACHE STRING "c flags")
set (CMAKE_CXX_FLAGS        "${CMAKE_CXX_FLAGS}" CACHE STRING "c++ flags")
SET (CMAKE_EXPORT_COMPILE_COMMANDS 1)
set (CMAKE_EXPORT_COMPILE_COMMANDS "${CMAKE_EXPORT_COMPILE_COMMANDS}" CACHE STRING "export compile_commands.json")

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

set(CMAKE_CXX_STANDARD 14)

find_package(Qt5Core REQUIRED)
include_directories(${Qt5Core_INCLUDE_DIRS})
add_definitions(${Qt5Core_DEFINITIONS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Qt5Core_EXECUTABLE_COMPILE_FLAGS}")


find_package(Qt5Test REQUIRED)
include_directories(${Qt5Test_INCLUDE_DIRS})

get_target_property(QtCore_Location Qt5::Core LOCATION)

MESSAGE("Using Qt version ${Qt5Core_VERSION} under ${QtCore_Location}")

set(FreeWill_SOURCES 
    Blob/Blob.cpp 
    FreeWillUnitTest.cpp)

add_executable(FreeWillUnitTest ${FreeWill_SOURCES})
target_link_libraries(FreeWillUnitTest Qt5::Core)
target_link_libraries(FreeWillUnitTest Qt5::Test)
