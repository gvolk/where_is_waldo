#ifndef GUI_CONTROLLER_H
#define GUI_CONTROLLER_H

#include "../src/data.h"
#include "mainwindow.h"
#include <QApplication>


enum training_state{
    SUBIMG, AREA1, AREA2, AREA3, TOP, BOT, FINISH
};

enum main_state{
    SELECT_WALDO, MARK_WALDO, FIND_WALDO, LOAD
};

class GuiController : public QObject
{
    Q_OBJECT
public:
    GuiController(int & argc, char ** argv);
    int run();
    void loadData(TrainingData, QList<QUrl>, QList<WaldoMarker>);
    ~GuiController();

public slots:
    void displayImage();
    void enterSubImage();
    void processPosition(bool, QPoint);
    void processMouseMoveEvent(QPoint);
    void processFinishState();
    void enterArea();
    void enterTopBottom();
    void processDroppedImates(QList<QUrl>);
    void processMenuAction(QAction *);

signals:
    void marked_waldos(QList<WaldoMarker>);
    void selected_waldo(TrainingData);
    void all_training_images_sig(QList<QUrl>);
    void find_waldo(QList<QUrl>, TrainingData);
    void load();


private:
    QApplication *app;
    MainWindow *window;
    void initSlots();
    void paintPoint(QPoint pos);
    void paintPath(QPainterPath *path, QPoint pos, QColor col);
    void fillPath(QPainterPath path, QColor col);
    void saveAreas();
    void enterFind();
    training_state train_state;
    main_state state;
    TrainingData data;
    QList<QUrl> all_training_images;
    QList<WaldoMarker> waldos;
    WaldoMarker current_waldo;
};


#endif // GUI_CONTROLLER_H

