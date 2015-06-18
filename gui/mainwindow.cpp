#include "mainwindow.h"
#include "ui_mainwindow.h"


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    setAcceptDrops(true);
    QObject::connect(ui->listWidget, SIGNAL(itemSelectionChanged()), this, SLOT(selectionChanged()));
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


void MainWindow::selectionChanged()
{
    QListWidgetItem *selected_item = ui->listWidget->currentItem();
    selected_item->text();
    qDebug() << "test";
    qDebug() << "Dropped file:" << selected_item->text();
    QPixmap image = QPixmap(selected_item->text());
    ui->label->setPixmap(image);
}

MainWindow::~MainWindow()
{
    delete ui;
}
