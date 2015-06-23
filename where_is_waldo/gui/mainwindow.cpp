#include "mainwindow.h"



MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    setAcceptDrops(true);

    scene = new QGraphicsScene();
    ui->graphicsView->setScene(scene);
    QObject::connect(scene, SIGNAL(), this, SLOT(processPosition(bool,QPoint)));
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
