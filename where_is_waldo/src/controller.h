#ifndef CONTROLLER_H
#define CONTROLLER_H

#include <vector>
#include <utility>
#include <fstream>
#include <ctime>

#include "camera_loading/CameraPoseData.h"
#include "camera_loading/CameraData.h"
#include "camera_loading/RotationQuaternion.hh"

#include "data.h"
#include "feature.h"
#include "dataprovider.h"
#include "../gui/guicontroller.h"
#include "classifier.h"

using namespace std;
using namespace base;

struct S {
    const char* path;
};

class Controller: public QObject
{
    Q_OBJECT

public:
    Controller(int & argc, char ** argv);
    int run();
    vector<S> LoadFilenamesFromFile(const char* filename);
    vector<pair<CameraDataf, CameraPoseDataf> > LoadCamerasFromFile(const char* filename);

    void compareGPUvsCPU(LogRegClassifier* c1,LogRegClassifier* c2,LogRegClassifier* c3, Feature* f );

    vector<QPoint> GetRefPoints(S s1, S s2, QPoint point);
    //float GetDiffFactor(QPoint top1, QPoint bottom1, QPoint top2, QPoint bottom2);

private:
    GuiController *gc;
    DataProvider *dp;
    Feature *f;
    LogRegClassifier* c1_class;
    LogRegClassifier* c2_class;
    LogRegClassifier* c3_class;

    void initSlots();

public slots:
    void save_marked_waldos(QList<WaldoMarker>);
    void save_selected_waldo(TrainingData);
    void save_all_images(QList<QUrl>);
    void search_waldo(QList<QUrl>, TrainingData*);
    void load();
};

#endif // CONTROLLER_H
