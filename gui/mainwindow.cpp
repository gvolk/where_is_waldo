#include "mainwindow.h"



MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    setAcceptDrops(true);

}


void MainWindow::dragEnterEvent(QDragEnterEvent *e)
{
    if (e->mimeData()->hasUrls()) {
        e->acceptProposedAction();
    }
}

void MainWindow::dropEvent(QDropEvent *e)
{
    foreach (const QUrl &url, e->mimeData()->urls()) {
        const QString &fileName = url.toLocalFile();

        ui->listWidget->addItem(fileName);
        qDebug() << "Dropped file:" << fileName;
    }
}



Ui::MainWindow* MainWindow::getMainWindow()
{
    return ui;
}

void MainWindow::setQPixmap(QPixmap *q)
{
    pixmap = q;
    updateAll();
}

QPixmap* MainWindow::getQPixmap()
{
    return pixmap;
}

void MainWindow::updateAll()
{
    ui->label->setPixmap(*pixmap);
}

MainWindow::~MainWindow()
{
    delete ui;
}
