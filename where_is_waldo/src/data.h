#ifndef DATA_H
#define DATA_H
#include <QApplication>
#include <QtWidgets>

#include "defines.h"

struct TrainingData{
    QUrl file;
    int orig_img_width;
    int orig_img_height;
    QPoint sub_img_start;
    int sub_img_width;
    int sub_img_heigth;
    QPoint top;
    QPoint bottom;
    QPainterPath area1;
    QPainterPath area2;
    QPainterPath area3;
};

struct WaldoMarker{
    QUrl file;
    QPoint sub_img_start;
    int sub_img_width;
    int sub_img_heigth;
};


#endif // DATA_H
