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
}

int GuiController::run()
{
    window->show();
    return app->exec();
}

void GuiController::displayImage()
{
    QListWidgetItem *selected_item = window->getMainWindow()->listWidget->currentItem();
    selected_item->text();
    int w = window->getMainWindow()->label->width();
    qDebug() << "show image:" << selected_item->text();
    QPixmap q(selected_item->text());
    window->setQPixmap(q.scaledToWidth(w));
}

void GuiController::markSubImage()
{
    qDebug() << "test";
    QPixmap q = window->getQPixmap();
    qDebug() << q;
    QPainter painter(&q);
    QPen Red((QColor(255,0,0)),5);
    painter.setPen(Red);
    painter.drawLine(50,50,1000,1000);
    qDebug() << "test2";
    window->setQPixmap(q);
    window->updateAll();
}

GuiController::~GuiController()
{
    delete app;
    delete window;
}



