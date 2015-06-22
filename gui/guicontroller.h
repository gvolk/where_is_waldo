#ifndef GUI_CONTROLLER_H
#define GUI_CONTROLLER_H

#include "mainwindow.h"
#include <QApplication>

#define REF_IMG "training_image.jpg"
#define REF_AREA "training_areas.jpg"

enum select_state{
    SUBIMG, AREA1, AREA2, AREA3, TOP, BOT, FINISH
};

struct Data{
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

class GuiController : public QObject
{
    Q_OBJECT
public:
    GuiController(int & argc, char ** argv);
    int run();
    ~GuiController();

private:
    QApplication *app;
    MainWindow *window;
    void initSlots();
    void paintPoint(QPoint pos);
    void paintPath(QPainterPath *path, QPoint pos, QColor col);
    void fillPath(QPainterPath path, QColor col);
    void saveAreas();
    select_state state;
    Data data;

public slots:
    void displayImage();
    void enterSubImage();
    void processPosition(bool, QPoint);
    void processMouseMoveEvent(QPoint);
    void processFinishState();
    void enterArea();
    void enterTopBottom();
};


#endif // GUI_CONTROLLER_H

