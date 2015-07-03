#include "controller.h"

Controller::Controller(int & argc, char ** argv)
{
    gc = new GuiController(argc, argv);
    dp = new DataProvider();
    initSlots();
}

int Controller::run()
{
    return gc->run();
}

void Controller::initSlots()
{
    QObject::connect(gc, SIGNAL(selected_waldo(TrainingData)), this, SLOT(save_selected_waldo(TrainingData)));
    QObject::connect(gc, SIGNAL(marked_waldos(QList<WaldoMarker>)), this, SLOT(save_marked_waldos(QList<WaldoMarker>)));
    QObject::connect(gc, SIGNAL(all_training_images_sig(QList<QUrl>)), this, SLOT(save_all_images(QList<QUrl>)));
    QObject::connect(gc, SIGNAL(find_waldo(QList<QUrl>,TrainingData*)), this, SLOT(search_waldo(QList<QUrl>,TrainingData*)));
    QObject::connect(gc, SIGNAL(load()), this, SLOT(load()));
}

void Controller::save_marked_waldos(QList<WaldoMarker> waldos)
{
    dp->saveMarkedTrainingData(waldos);
}

void Controller::save_all_images(QList<QUrl> images)
{
    dp->saveAllImages(images);
}

void Controller::save_selected_waldo(TrainingData data)
{
    dp->saveSelectedWaldo(data);
}

void Controller::search_waldo(QList<QUrl> urls, TrainingData *data)
{
    f = new Feature(data);
    f->createFeatures();
    c_class = new CPU_Classifier();

    for(int i=0; i<9; i++)
    {
        qDebug() << f->getFeature(1)->features[i];
    }
    qDebug() << f->getFeature(1)->labels[0];

    c_class->train(f->getFeature(1));
    c_class->test_classification(f->getFeature(1), f->getFeature(1));

    //TODO
}

void Controller::load()
{
    gc->loadData(dp->loadSelectedWaldo(), dp->loadAllImages(), dp->loadMarkedTrainingData());
}

