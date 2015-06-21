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
    QObject::connect(window->getMainWindow()->subImageButton, SIGNAL(clicked()), this, SLOT(enterSubImage()));
    QObject::connect(window->getMainWindow()->saveButton, SIGNAL(clicked()), this, SLOT(processFinishState()));
    QObject::connect(window->getMainWindow()->graphicsView, SIGNAL(posSignal(bool,QPoint)), this, SLOT(processPosition(bool,QPoint)));
    QObject::connect(window->getMainWindow()->graphicsView, SIGNAL(filteredMouseEvent(QPoint)), this, SLOT(processMouseMoveEvent(QPoint)));
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

void GuiController::enterSubImage()
{
    state = SUBIMG;
    window->getMainWindow()->graphicsView->resetSelectionStart();
}

void GuiController::processMouseMoveEvent(QPoint pos)
{
    QPixmap q = window->getOrigQPixmap();
    QPainter painter(&q);
    QPen Red((QColor(255,0,0)),3);
    painter.setPen(Red);
    int width = pos.x() - data.sub_img_start.x();
    int height = pos.y() - data.sub_img_start.y();
    painter.drawRect(data.sub_img_start.x(), data.sub_img_start.y(), width, height);
    window->setQPixmap(q);
    window->updateAll();
}

void GuiController::processPosition(bool started, QPoint pos)
{
    switch(state)
    {
        case SUBIMG:
            {
                if(started)
                {
                    data.sub_img_start.setX(pos.x());
                    data.sub_img_start.setY(pos.y());
                }
                else
                {
                    data.sub_img_width = pos.x() - data.sub_img_start.x();
                    data.sub_img_heigth = pos.y() - data.sub_img_start.y();
                }
                break;
            }
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

void GuiController::processFinishState()
{
    switch(state)
    {
        case SUBIMG:
            {
                QRect rect(data.sub_img_start.x(), data.sub_img_start.y(), data.sub_img_width, data.sub_img_heigth);
                QPixmap cropped = window->getOrigQPixmap().copy(rect);

                qDebug() << QDir::currentPath();
                QFile file(REF_IMG);
                file.open(QIODevice::WriteOnly);
                file.write("test");
                cropped.save(&file, "JPG");
                file.flush();
                file.close();



                window->setQPixmap(cropped);
                window->updateAll();
                window->getMainWindow()->topButton->setChecked(true);
                state = TOPBOT;
                break;
            }
        case TOPBOT:
            //TODO
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



