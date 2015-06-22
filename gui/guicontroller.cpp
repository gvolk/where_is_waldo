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
    QObject::connect(window->getMainWindow()->areaButton, SIGNAL(clicked()), this, SLOT(enterArea()));
    QObject::connect(window->getMainWindow()->topButton, SIGNAL(clicked()), this, SLOT(enterTopBottom()));
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
    window->getMainWindow()->plainTextEdit->setPlainText("mark subimage with mouse");
    data.orig_img_height = window->getQPixmap().height();
    data.orig_img_width = window->getQPixmap().width();
    window->getMainWindow()->topButton->setEnabled(false);
    window->getMainWindow()->areaButton->setEnabled(false);
    window->getMainWindow()->subImageButton->setEnabled(true);
    window->getMainWindow()->graphicsView->resetSelectionStart();
    state = SUBIMG;

}

void GuiController::enterArea()
{
    window->setQPixmap(QPixmap(REF_IMG));
    window->getMainWindow()->plainTextEdit->setPlainText("select most important area of waldo");
    window->getMainWindow()->areaButton->setEnabled(true);
    window->getMainWindow()->topButton->setEnabled(false);
    window->getMainWindow()->areaButton->setChecked(true);
    window->getMainWindow()->graphicsView->resetSelectionStart();
    state = AREA1;
}

void GuiController::enterTopBottom()
{
    window->getMainWindow()->plainTextEdit->setPlainText("mark the top of the person");
    window->getMainWindow()->topButton->setEnabled(true);
    window->getMainWindow()->topButton->setChecked(true);
    state = TOP;
}

void GuiController::processMouseMoveEvent(QPoint pos)
{

    if(state == SUBIMG)
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
    else if(state == AREA1 )
    {
        paintPath(&data.area1, pos, QColor(255,0,0,128));
    }
    else if(state == AREA2 )
    {
        paintPath(&data.area2, pos, QColor(0,255,0,128));
    }
    else if(state == AREA3 )
    {
        paintPath(&data.area3, pos, QColor(0,0,255,128));
    }
}

void GuiController::paintPath(QPainterPath *path, QPoint pos, QColor col)
{
    QPixmap q = window->getQPixmap();
    QPainter painter(&q);
    path->lineTo(pos);
    QPen Red(col,3);
    painter.setPen(Red);
    painter.drawPath(*path);
    window->setQPixmap(q);
    window->updateAll();
}

void GuiController::paintPoint(QPoint pos)
{
        QPixmap q = window->getQPixmap();
        QPainter painter(&q);
        QPen Red((QColor(255,0,0)),7);
        painter.setPen(Red);
        painter.drawPoint(pos.x(),pos.y());
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
        case TOP:
        {
            if(started)
                data.top = pos;
                window->getMainWindow()->plainTextEdit->setPlainText("mark the bottom of the person");
                state = BOT;
                paintPoint(pos);
            break;
        }
        case BOT:
        {
            if(started)
                data.bottom = pos;
                paintPoint(pos);
            break;
        }
        case AREA1:
        {
            data.area1.moveTo(pos);
            break;
        }
        case AREA2:
        {
            data.area2.moveTo(pos);
            break;
        }
        case AREA3:
        {
            data.area3.moveTo(pos);
            break;
        }
        default:
            //nothing to do
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
                cropped.save(&file, "JPG");

                window->setQPixmap(cropped);
                window->updateAll();
                window->getMainWindow()->topButton->setChecked(true);
                enterTopBottom();
                break;
            }
        case BOT:
        {
            enterArea();
            break;
        }
        case AREA1:
        {
            fillPath(data.area1, QColor(255,0,0,128));
            window->getMainWindow()->plainTextEdit->setPlainText("select scecond most important area of waldo");
            state = AREA2;
            break;
        }
        case AREA2:
        {
            fillPath(data.area2, QColor(0,255,0,128));
            window->getMainWindow()->plainTextEdit->setPlainText("select third most important area of waldo");
            state = AREA3;
            break;
        }
        case AREA3:
        {
            fillPath(data.area3, QColor(0,0,255,128));
            window->getMainWindow()->plainTextEdit->setPlainText("finish to save selected areas");
            state = FINISH;
            break;
        }
        case FINISH:
        {
            window->getMainWindow()->plainTextEdit->setPlainText("selected areas saved");
            saveAreas();
        }
        default:
            //nothing to do
            break;
    }
}


void GuiController::fillPath(QPainterPath path, QColor col)
{
    QPixmap q = window->getQPixmap();
    QPainter painter(&q);
    painter.setPen(Qt::NoPen);
    painter.fillPath(path, QBrush(col));
    window->setQPixmap(q);
    window->updateAll();
}

void GuiController::saveAreas()
{
    QPixmap q(REF_IMG);
    QPixmap areas( q.width(), q.height());
    QPainter painter(&areas);

    painter.setPen(Qt::NoPen);
    QRect rect(0, 0, q.width(), q.height());
    painter.fillRect(rect,QBrush(QColor(0,0,0)));

    painter.fillPath(data.area1, QBrush(QColor(255,0,0)));
    painter.fillPath(data.area2, QBrush(QColor(0,255,0)));
    painter.fillPath(data.area3, QBrush(QColor(0,0,255)));
    QFile file(REF_AREA);
    file.open(QIODevice::WriteOnly);
    areas.save(&file, "JPG");
}



GuiController::~GuiController()
{
    delete app;
    delete window;
}



