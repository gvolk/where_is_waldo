#ifndef GUI_CONTROLLER_H
#define GUI_CONTROLLER_H

#include "mainwindow.h"
#include <QApplication>

#define REF_IMG "training_image.jpg"
#define REF_AREA "training_areas.jpg"

enum select_state{
    SUBIMG, AREA1, AREA2, AREA3, TOPBOT
};

struct Data{
    QPoint sub_img_start;
    int sub_img_width;
    int sub_img_heigth;
    QPoint top;
    QPoint bottom;
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
    select_state state;
    Data data;

public slots:
    void displayImage();
    void enterSubImage();
    void processPosition(bool, QPoint);
    void processMouseMoveEvent(QPoint);
    void processFinishState();


};


#endif // GUI_CONTROLLER_H

