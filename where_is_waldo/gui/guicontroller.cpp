#include "guicontroller.h"

GuiController::GuiController(int & argc, char ** argv)
{
    app = new QApplication(argc, argv);
    setlocale(LC_NUMERIC,"C");
    window = new MainWindow();


    window->getMainWindow()->subImageButton->setChecked(true);
    state = SELECT_WALDO;
    train_state = SUBIMG;

    initSlots();
}

void GuiController::initSlots()
{
    QObject::connect(window->getMainWindow()->listWidget, SIGNAL(itemSelectionChanged()), this, SLOT(displayImage()));
    QObject::connect(window->getMainWindow()->listWidget_marked, SIGNAL(itemSelectionChanged()), this, SLOT(displayMarkedWaldo()));
    QObject::connect(window->getMainWindow()->listWidget_found, SIGNAL(itemSelectionChanged()), this, SLOT(displayFoundWaldo()));
    QObject::connect(window->getMainWindow()->subImageButton, SIGNAL(clicked()), this, SLOT(enterSubImage()));
    QObject::connect(window->getMainWindow()->areaButton, SIGNAL(clicked()), this, SLOT(enterArea()));
    QObject::connect(window->getMainWindow()->topButton, SIGNAL(clicked()), this, SLOT(enterTopBottom()));
    QObject::connect(window->getMainWindow()->saveButton, SIGNAL(clicked()), this, SLOT(processFinishState()));
    QObject::connect(window->getMainWindow()->graphicsView, SIGNAL(posSignal(bool,QPoint)), this, SLOT(processPosition(bool,QPoint)));
    QObject::connect(window->getMainWindow()->graphicsView, SIGNAL(filteredMouseEvent(QPoint)), this, SLOT(processMouseMoveEvent(QPoint)));
    QObject::connect(window, SIGNAL(droppedUrls(QList<QUrl>)) , this, SLOT(processDroppedImates(QList<QUrl>)));
    QObject::connect(window->getMainWindow()->menuBar, SIGNAL(triggered(QAction*)) , this, SLOT(processMenuAction(QAction*)));
}

int GuiController::run()
{
    window->show();
    return app->exec();
}

void GuiController::processMenuAction(QAction *action)
{
    if(window->getMainWindow()->actionSelect_waldo == action)
    {
        state = SELECT_WALDO;
        enterSubImage();
    }
    else if(window->getMainWindow()->actionMark_waldo == action)
    {
        state = MARK_WALDO;
        enterSubImage();
    }
    else if(window->getMainWindow()->actionFind_waldo == action)
    {
        enterFind();
    }
    else if(window->getMainWindow()->actionLoad_training_data == action)
    {
        window->getMainWindow()->saveButton->setText("load");
        state = LOAD;
    }
}


void GuiController::processDroppedImates(QList<QUrl> urls)
{
    QDir dir = QDir::currentPath();
    all_training_images.clear();
    foreach (const QUrl &url, urls) {
        const QUrl &image = dir.relativeFilePath(url.toLocalFile());
        all_training_images.append(image);
    }

    window->updateList(all_training_images);
}

void GuiController::markedWaldosChanged()
{
    QList<QUrl> *list = new QList<QUrl>();
    foreach (const WaldoMarker &waldo, waldos) {
        list->append(QUrl(waldo.file.toString()));
    }
    window->updateMarkedList(*list);
}

void GuiController::foundWaldosChanged()
{
    QList<QUrl> *list = new QList<QUrl>();
    foreach (const WaldoMarker &waldo, found_waldos) {
        list->append(QUrl(waldo.file.toString()));
    }
    window->updateFoundList(*list);
}

void GuiController::displayImage()
{
    if(state == SELECT_WALDO || state == MARK_WALDO)
    {
        window->setQPixmap(window->getOrigQPixmap());
    }
    else if(state == FIND_WALDO)
    {
        QPixmap q(REF_IMG);
        window->setQPixmap(q);
        fillPath(data.area1, QColor(255,0,0,128));
        fillPath(data.area2, QColor(0,255,0,128));
        fillPath(data.area3, QColor(0,0,255,128));
        window->updateAll();
    }
}

void GuiController::displayMarkedWaldo()
{
    if(window->getMainWindow()->listWidget_marked->selectedItems().isEmpty())
    {
        return;
    }
    QString img = window->getMainWindow()->listWidget_marked->selectedItems().first()->text();
    QPixmap q(img);
    QPainter painter(&q);
    QPen Red((QColor(255,0,0)),3);
    painter.setPen(Red);
    foreach (const WaldoMarker &waldo, waldos) {
        if(waldo.file.toString() == img)
        {
            painter.drawRect(waldo.sub_img_start.x(), waldo.sub_img_start.y(), waldo.sub_img_width, waldo.sub_img_heigth);
        }
    }

    window->setQPixmap(q);
    window->updateAll();
}

void GuiController::displayFoundWaldo()
{
    if(window->getMainWindow()->listWidget_found->selectedItems().isEmpty())
    {
        return;
    }

    QString img = window->getMainWindow()->listWidget_found->selectedItems().first()->text();
    QPixmap q(img);
    QPainter painter(&q);
    QPen Red((QColor(255,0,0)),3);
    painter.setPen(Red);
    foreach (const WaldoMarker &waldo, found_waldos) {
        if(waldo.file.toString() == img)
        {
            painter.drawRect(waldo.sub_img_start.x(), waldo.sub_img_start.y(), waldo.sub_img_width, waldo.sub_img_heigth);
        }
    }

    window->setQPixmap(q);
    window->updateAll();
}

void GuiController::addFoundWaldo(WaldoMarker waldo)
{
    found_waldos.append(waldo);
    foundWaldosChanged();
}

void GuiController::clearFoundWaldo()
{
    found_waldos.clear();
    foundWaldosChanged();
}

void GuiController::enterFind()
{
    window->statusBar()->showMessage("Search for waldo in List of Images.");
    window->getMainWindow()->topButton->setEnabled(false);
    window->getMainWindow()->areaButton->setEnabled(false);
    window->getMainWindow()->subImageButton->setEnabled(false);
    window->getMainWindow()->saveButton->setText("search");
    state = FIND_WALDO;
}

void GuiController::enterSubImage()
{
    if(state == SELECT_WALDO)
    {
        window->statusBar()->showMessage("mark subimage with mouse");
        train_state = SUBIMG;
    }
    else if (state == MARK_WALDO)
    {
        window->statusBar()->showMessage("mark waldo on images");
    }

    window->getMainWindow()->topButton->setEnabled(false);
    window->getMainWindow()->areaButton->setEnabled(false);
    window->getMainWindow()->subImageButton->setEnabled(true);
    window->getMainWindow()->graphicsView->resetSelectionStart();
    window->getMainWindow()->saveButton->setText("save");

}

void GuiController::enterArea()
{
    window->setQPixmap(QPixmap(REF_IMG));
    window->statusBar()->showMessage("select most important area of waldo");
    window->getMainWindow()->areaButton->setEnabled(true);
    window->getMainWindow()->topButton->setEnabled(false);
    window->getMainWindow()->areaButton->setChecked(true);
    window->getMainWindow()->graphicsView->resetSelectionStart();
    train_state = AREA1;
}

void GuiController::enterTopBottom()
{
    window->statusBar()->showMessage("mark the top of the person");
    window->getMainWindow()->topButton->setEnabled(true);
    window->getMainWindow()->topButton->setChecked(true);
    train_state = TOP;
}

void GuiController::processMouseMoveEvent(QPoint pos)
{
    if(window->getMainWindow()->listWidget->selectedItems().isEmpty())
    {
        window->statusBar()->showMessage("please select Image before marking area");
        return;
    }
    if( state == SELECT_WALDO)
    {
        if(train_state == SUBIMG)
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
        else if(train_state == AREA1 )
        {
            paintPath(&data.area1, pos, QColor(255,0,0,128));
        }
        else if(train_state == AREA2 )
        {
            paintPath(&data.area2, pos, QColor(0,255,0,128));
        }
        else if(train_state == AREA3 )
        {
            paintPath(&data.area3, pos, QColor(0,0,255,128));
        }
    }
    else if(state == MARK_WALDO)
    {
        QPixmap q = window->getOrigQPixmap();
        QPainter painter(&q);
        QPen Red((QColor(255,0,0)),3);
        painter.setPen(Red);
        int width = pos.x() - current_waldo.sub_img_start.x();
        int height = pos.y() - current_waldo.sub_img_start.y();
        painter.drawRect(current_waldo.sub_img_start.x(), current_waldo.sub_img_start.y(), width, height);
        window->setQPixmap(q);
        window->updateAll();
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
    if(window->getMainWindow()->listWidget->selectedItems().isEmpty())
    {
        window->statusBar()->showMessage("please select Image before marking area");
        return;
    }
    if(state == SELECT_WALDO)
    {
        switch(train_state)
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
                    window->statusBar()->showMessage("mark the bottom of the person");
                    train_state = BOT;
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
    else if(state == MARK_WALDO)
    {
        if(started)
        {
            current_waldo.sub_img_start.setX(pos.x());
            current_waldo.sub_img_start.setY(pos.y());
        }
        else
        {
            current_waldo.sub_img_width = pos.x() - current_waldo.sub_img_start.x();
            current_waldo.sub_img_heigth = pos.y() - current_waldo.sub_img_start.y();
        }
    }
}

void GuiController::processFinishState()
{
    if(state == SELECT_WALDO)
    {
        switch(train_state)
        {
            case SUBIMG:
                {
                    //save original heigth and width of image
                    data.orig_img_height = window->getQPixmap().height();
                    data.orig_img_width = window->getQPixmap().width();

                    QRect rect(data.sub_img_start.x(), data.sub_img_start.y(), data.sub_img_width, data.sub_img_heigth);
                    QPixmap cropped = window->getOrigQPixmap().copy(rect);

                    QFile file(REF_IMG);
                    file.open(QIODevice::WriteOnly);
                    //cropped.save(&file, "JPG");
                    cropped.save(&file, "PPM");

                    QUrl orig_file = QUrl(window->getMainWindow()->listWidget->selectedItems().first()->text());
                    data.file = orig_file;

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
                window->statusBar()->showMessage("select scecond most important area of waldo");
                train_state = AREA2;
                break;
            }
            case AREA2:
            {
                fillPath(data.area2, QColor(0,255,0,128));
                window->statusBar()->showMessage("select third most important area of waldo");
                train_state = AREA3;
                break;
            }
            case AREA3:
            {
                fillPath(data.area3, QColor(0,0,255,128));
                window->statusBar()->showMessage("finish to save selected areas");
                train_state = FINISH;
                break;
            }
            case FINISH:
            {
                emit selected_waldo(data);
                emit all_training_images_sig(all_training_images);
                window->statusBar()->showMessage("selected areas saved");
                saveAreas();
                waldos.clear();
            }
            default:
                //nothing to do
                break;
        }
    }
    else if(state == MARK_WALDO)
    {
        QUrl file = QUrl(window->getMainWindow()->listWidget->selectedItems().first()->text());
        current_waldo.file = file;

        waldos.append(current_waldo);
        current_waldo = WaldoMarker();
        markedWaldosChanged();

        emit marked_waldos(waldos);
    }
    else if(state == FIND_WALDO)
    {
        clearFoundWaldo();
        emit find_waldo(all_training_images, &data);
        checkFoundWaldos();
    }
    else if(state == LOAD)
    {
        emit load();
    }
}

void GuiController::loadData(TrainingData new_data, QList<QUrl> new_images, QList<WaldoMarker> new_waldos)
{
    data = new_data;
    all_training_images = new_images;
    waldos = new_waldos;

    window->updateList(all_training_images);
    markedWaldosChanged();

    displayImage();
    enterFind();
    //enterSubImage();
}

void GuiController::checkFoundWaldos()
{
    int num_waldos = waldos.length();
    int num_found_waldos = found_waldos.length();
    QString output = QString("Found %1 out of %2 marked waldos").arg(num_found_waldos).arg(num_waldos);
    window->statusBar()->showMessage(output);
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
    //areas.save(&file, "JPG");
    areas.save(&file, "PPM");
}



GuiController::~GuiController()
{
    delete app;
    delete window;
}



