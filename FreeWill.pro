TEMPLATE = subdirs
CONFIG   += c++14
SUBDIRS = FreeWill \
    Demos/TicTacToe \
    Demos/Word2Vec

TicTacToe.depends = FreeWill
Word2Vec.depends = FreeWill
