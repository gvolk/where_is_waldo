#include "mygraphicsview.h"

MyGraphicsView::MyGraphicsView(QObject *parent)
{
    selection_started = true;
    timestamp = 0;
}

void MyGraphicsView::mousePressEvent(QMouseEvent *e)
{
    qDebug() << "mouse button pressed";
    if(selection_started == true)
    {
        emit posSignal(selection_started, includeScrollbarOffset(e->pos()));
        selection_started = false;
    }
    else
    {
        qDebug() << "Selection started should be true";
    }
}

void MyGraphicsView::mouseMoveEvent(QMouseEvent *e)
{
    if(timestamp == 0)
    {
        timestamp = e->timestamp();
        return;
    }
    ulong diff = e->timestamp() -timestamp;

    //only process mouse event if selection started and 50ms since last event
    if(selection_started == false && diff > 50)
    {
        emit filteredMouseEvent(includeScrollbarOffset(e->pos()));
        timestamp = 0;
    }
}

void MyGraphicsView::mouseReleaseEvent(QMouseEvent *e)
{
    qDebug() << "mouse button released";
    if(selection_started == false)
    {
        emit posSignal(selection_started, includeScrollbarOffset(e->pos()));
        selection_started = true;
    }
    else
    {
        qDebug() << "Selection started should be false";
    }
}

//add the offset by the scrollbar to get the correct coordinates
QPoint MyGraphicsView::includeScrollbarOffset(QPoint point)
{
    int new_x = point.x() + horizontalScrollBar()->value();
    int new_y = point.y() + verticalScrollBar()->value();
    QPoint newpoint(new_x, new_y);
    return newpoint;
}

void MyGraphicsView::resetSelectionStart()
{
    selection_started = true;
}
