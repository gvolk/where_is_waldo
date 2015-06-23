#ifndef MYGRAPHICSVIEW_H
#define MYGRAPHICSVIEW_H

#include <QMainWindow>
#include <QtWidgets>
#include <QDragEnterEvent>
#include <QDropEvent>

class MyGraphicsView : public QGraphicsView
{
    Q_OBJECT
public:
    MyGraphicsView(QObject *parent = 0);
    void resetSelectionStart();
private:
    void mousePressEvent(QMouseEvent *e);
    void mouseMoveEvent(QMouseEvent *e);
    void mouseReleaseEvent(QMouseEvent *e);
    QPoint includeScrollbarOffset(QPoint);
    bool selection_started;
    ulong timestamp;

signals:
    //bool shows wheter start or end point
    void posSignal(bool,QPoint);
    void filteredMouseEvent(QPoint);

public slots:



};

#endif // MYGRAPHICSVIEW_H
