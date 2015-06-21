#-------------------------------------------------
#
# Project created by QtCreator 2015-06-16T21:13:07
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = gui
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp \
    guicontroller.cpp \
    mygraphicsview.cpp

HEADERS  += mainwindow.h \
    guicontroller.h \
    mygraphicsview.h

FORMS    += mainwindow.ui
