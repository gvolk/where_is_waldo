#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QtWidgets>
#include <QDragEnterEvent>
#include <QDropEvent>
#include "ui_mainwindow.h"


namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    Ui::MainWindow* getMainWindow();
    void setQPixmap(QPixmap);
    QPixmap  getQPixmap();
    void updateAll();
    ~MainWindow();


private:
    void dragEnterEvent(QDragEnterEvent *e);
    void dropEvent(QDropEvent *e);
    Ui::MainWindow *ui;
    QPixmap pixmap;

};

#endif // MAINWINDOW_H
