#ifndef DATA_H
#define DATA_H
#include <QApplication>
#include <QtWidgets>

//#define REF_IMG "training_image.jpg"
#define REF_IMG "training_image.ppm"
//#define REF_AREA "training_areas.jpg"
#define REF_AREA "training_areas.ppm"

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
