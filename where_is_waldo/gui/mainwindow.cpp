#include "mainwindow.h"



MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    setAcceptDrops(true);

    scene = new QGraphicsScene();
    ui->graphicsView->setScene(scene);
}


void MainWindow::dragEnterEvent(QDragEnterEvent *e)
{
    if (e->mimeData()->hasUrls()) {
        e->acceptProposedAction();
    }
}

void MainWindow::dropEvent(QDropEvent *e)
{
    emit droppedUrls(e->mimeData()->urls());
}

void MainWindow::updateList(QList <QUrl> urls)
{
    ui->listWidget->clear();
    foreach (const QUrl &url, urls) {
        const QString &fileName = url.toString();
        ui->listWidget->addItem(fileName);
    }
}

void MainWindow::updateMarkedList(QList <QUrl> urls)
{
    ui->listWidget_marked->clear();
    foreach (const QUrl &url, urls) {
        const QString &fileName = url.toString();
        ui->listWidget_marked->addItem(fileName);
    }
}

void MainWindow::updateFoundList(QList <QUrl> urls)
{
    ui->listWidget_found->clear();
    foreach (const QUrl &url, urls) {
        const QString &fileName = url.toString();
        ui->listWidget_found->addItem(fileName);
    }
}

Ui::MainWindow* MainWindow::getMainWindow()
{
    return ui;
}

void MainWindow::setQPixmap(QPixmap q)
{

    pixmap = q;
    updateAll();
}

QPixmap MainWindow::getQPixmap()
{
    return pixmap;
}

QPixmap MainWindow::getOrigQPixmap()
{
    QListWidgetItem *selected_item = ui->listWidget->currentItem();
    QPixmap q(selected_item->text());
    return q;
}

void MainWindow::updateAll()
{
    scene->clear();
    QGraphicsPixmapItem *item = new QGraphicsPixmapItem(pixmap);
    scene->addItem(item);
    ui->graphicsView->show();
}

MainWindow::~MainWindow()
{
    delete ui;
}
