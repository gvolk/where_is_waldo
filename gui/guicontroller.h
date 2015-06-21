#ifndef GUI_CONTROLLER_H
#define GUI_CONTROLLER_H

#include "mainwindow.h"
#include <QApplication>

enum select_state{
    SUBIMG, AREA1, AREA2, AREA3, TOPBOT
};

struct Data{
    QPoint sub_img_start;
    QPoint sub_img_end;
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
    void markSubImage();
    void processPosition(bool, QPoint);
    void processMouseMoveEvent(QMouseEvent*);


};


#endif // GUI_CONTROLLER_H

