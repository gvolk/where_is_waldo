#ifndef DATA_H
#define DATA_H
#include <QApplication>
#include <QtWidgets>

struct TrainingData{
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


#endif // DATA_H
