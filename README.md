# FreeWill

![futurama freewill unit](splash.jpg)

![doc build badge](https://readthedocs.org/projects/freewill/badge/?version=latest)

FreeWill is a deeplearning library implemented in C++. The purpose of writing FreeWill is for me to understand deeplearning in detail. In addition to the library itself, I will try to write detailed document and examples.

The first goal of this project is matching https://github.com/karpathy/convnetjs feature wise.

= How to build =
So far, I have only tested on Ubuntu 16

I have been avoiding introducing dependencies to this project, but you need to have Qt5 to use it.

Make sure you have also installed Qt5Websockets, this may not be included in Qt5Base

    sudo apt-get install libqt5websockets5-dev

To build:

    mkdir build
    cd build
    cmake .. -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=debug

In case you want to specify which Qt5 to use (for example you want to build Qt yourself), do this to config this project:

    CMAKE_PREFIX_PATH=/home/shiy/Qt/5.7/gcc_64 cmake ..  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=debug


