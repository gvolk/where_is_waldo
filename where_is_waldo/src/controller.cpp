#include "controller.h"

using namespace std;
using namespace base;

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
    /**
     * the following part only work for a special set of picture.
     * these are the pictures in the image dir of the project.
     * these special pictures have been put in the program VisualFM
     * and there a 3D scene of these picture was created.
     * All positioning information of these pictures are saved in a
     * .nvm-file which is also placed in the project (and is static).
     *
     * So to include other pictures it would be necessary to put them
     * in VisualFM and create another 3D scene.
     *
     * VisualFM wasnt able to position all of our 38 pictures, so only
     * 17 pictures were positioned and only in these 17 pictures, we
     * are able to search for waldo.
     *
     * NOTE: Some of the following code is from intern projects of
     * Prof. Dr. Lensch and his team and isnt public.
     * So these files are not included in any public git repositories.
     *
     */


}

struct S {
    char path[512];
};

vector<S> LoadFilenamesFromFile(const char* filename) {
    vector<S> result;

    ifstream fin(filename);
    if (!fin.is_open())
        exit(1);

    string line;
    getline(fin, line);

    if (!line.compare("NVM_V3"))
        exit(1);

    getline(fin, line);
    getline(fin, line);
    unsigned numImages = atoi(line.c_str());
    for (unsigned int i = 0; i < numImages; i++) {
        getline(fin, line);

        S s;

        // only path is relevant.
        float tmp;
        sscanf(line.c_str(), "%s	%f %f %f %f %f %f %f %f", s.path, &tmp,
                        &tmp, &tmp, &tmp, &tmp, &tmp, &tmp, &tmp);
        // now save path

        result.push_back(s);
    }
    fin.close();

    return result;
}

vector<pair<CameraDataf, CameraPoseDataf> > LoadCamerasFromFile(
        const char* filename)
{
    vector<pair<CameraDataf, CameraPoseDataf> > result;

    ifstream fin(filename);
    if (!fin.is_open())
        exit(1);

    string line;
    getline(fin, line);

    if (!line.compare("NVM_V3"))
        exit(1);

    getline(fin, line);
    getline(fin, line);
    unsigned numImages = atoi(line.c_str());
    for (unsigned int i = 0; i < numImages; i++)
    {
        getline(fin, line);
        char path[512];
        RotationQuaternion<float> rotation;
        Vec3f position;
        float focalLength;
        sscanf(line.c_str(), "%s	%f %f %f %f %f %f %f %f", path, &focalLength,
                &rotation[3], &rotation[0], &rotation[2], &rotation[1],
                &position[0], &position[2], &position[1]);

        CameraDataf intrinsics;
        intrinsics.FocalLength[0] = focalLength;
        intrinsics.FocalLength[1] = focalLength;

        QPixmap q(path);
        //intrinsics.ImageSize = Vec2f(2848, 2136); // TODO: Set this correctly from image
        intrinsics.ImageSize = Vec2f(q.width(), q.height());

        CameraPoseDataf extrinsics;
        extrinsics.Alignment = rotation.GetRotationMatrix();
        for (unsigned int d = 0; d < 3; d++)
            extrinsics.Alignment(d, 3) = position[d];

        result.push_back(make_pair(intrinsics, extrinsics));
    }
    fin.close();

    return result;
}

void Controller::load()
{
    gc->loadData(dp->loadSelectedWaldo(), dp->loadAllImages(), dp->loadMarkedTrainingData());
}

