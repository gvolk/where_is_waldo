#include "guicontroller.h"

GuiController::GuiController(int & argc, char ** argv)
{
    app = new QApplication(argc, argv);
    window = new MainWindow();

    initSlots();
}

void GuiController::initSlots()
{
    QObject::connect(window->getMainWindow()->listWidget, SIGNAL(itemSelectionChanged()), this, SLOT(displayImage()));
    QObject::connect(window->getMainWindow()->subImageButton, SIGNAL(clicked()), this, SLOT(markSubImage()));
    QObject::connect(window->getMainWindow()->graphicsView, SIGNAL(posSignal(bool,QPoint)), this, SLOT(processPosition(bool,QPoint)));
    QObject::connect(window->getMainWindow()->graphicsView, SIGNAL(filteredMouseEvent(QMouseEvent*)), this, SLOT(processMouseMoveEvent(QMouseEvent*)));
}

int GuiController::run()
{
    window->show();
    return app->exec();
}

void GuiController::displayImage()
{
    window->setQPixmap(window->getOrigQPixmap());
}

void GuiController::markSubImage()
{
    state = SUBIMG;
    window->getMainWindow()->graphicsView->resetSelectionStart();
}

void GuiController::processMouseMoveEvent(QMouseEvent *e)
{
    QPixmap q = window->getOrigQPixmap();
    QPainter painter(&q);
    QPen Red((QColor(255,0,0)),3);
    painter.setPen(Red);
    int width = e->pos().x() - data.sub_img_start.x();
    int height = e->pos().y() - data.sub_img_start.y();
    painter.drawRect(data.sub_img_start.x(), data.sub_img_start.y(), width, height);
    window->setQPixmap(q);
    window->updateAll();
}

void GuiController::processPosition(bool started, QPoint pos)
{

    switch(state)
    {
        case SUBIMG:
            if(started)
            {
                data.sub_img_start.setX(pos.x());
                data.sub_img_start.setY(pos.y());
            }
            else
            {
                data.sub_img_end.setX(pos.x());
                data.sub_img_end.setY(pos.y());
            }
            break;
        case TOPBOT:
            if(started)
                data.top = pos;
            else
                data.bottom = pos;
            break;
        case AREA1:
            //TODO
            break;
        case AREA2:
            //TODO
            break;
        case AREA3:
            //TODO
            break;
    }
}

GuiController::~GuiController()
{
    delete app;
    delete window;
}



