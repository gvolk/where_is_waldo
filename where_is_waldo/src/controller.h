#ifndef CONTROLLER_H
#define CONTROLLER_H

#include "data.h"
#include "feature.h"
#include "dataprovider.h"
#include "../gui/guicontroller.h"
#include "cpu_classifier.h"

class Controller: public QObject
{
    Q_OBJECT

public:
    Controller(int & argc, char ** argv);
    int run();

private:
    GuiController *gc;
    DataProvider *dp;
    Feature *f;
    CPU_Classifier* c_class;
    void initSlots();

public slots:
    void save_marked_waldos(QList<WaldoMarker>);
    void save_selected_waldo(TrainingData);
    void save_all_images(QList<QUrl>);
    void search_waldo(QList<QUrl>, TrainingData*);
    void load();
};

#endif // CONTROLLER_H
