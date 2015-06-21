#ifndef GUI_CONTROLLER_H
#define GUI_CONTROLLER_H

#include "mainwindow.h"
#include <QApplication>


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

public slots:
    void displayImage();
    void markSubImage();

};


#endif // GUI_CONTROLLER_H

