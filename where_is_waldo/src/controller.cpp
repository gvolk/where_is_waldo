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
    qDebug() << data.area1;
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
    string tmp = "camera_loading/test.nvm";
    const char* filename = tmp.c_str();

    // Load file paths.
    vector<S> paths = LoadFilenamesFromFile(filename);

    // Load camera data.
    vector<pair<CameraDataf, CameraPoseDataf> > cameraData =
                LoadCamerasFromFile(filename);

    // Pick to cameras
    CameraDataf cam1 = cameraData[0].first;
    CameraPoseDataf pose1 = cameraData[0].second;
    CameraDataf cam2 = cameraData[12].first;
    CameraPoseDataf pose2 = cameraData[12].second;

    // === Project a point from one camera to the other ===
    Vec2f image1Point(cam1.ImageSize / 2); // Pick a point at the center of the image

    // Process several depths
    for (float d = 0.4f; d <= 1.6f; d += 0.3f)
    {
        // Transform point to viewspace vector
        Vec3f cam1Dir = cam1.GetViewspaceDirection(image1Point);
        // Transform to worldspace ray
        Rayf worldRay = pose1.GetWorldspaceRay(cam1Dir);

        // Pick a point on the ray according to depth
        Vec3f worldPoint = worldRay.Origin + worldRay.Direction * d;
        // ATTENTION: The depth is interpreted along the normalized ray direction, not along the principal camera axis!
        // Maybe you have to change this.

        // Transform to second camera's viewspace
        Vec3f cam2Dir = pose2.GetViewspaceDirectionFromPoint(worldPoint);
        // Transform to second image
        Vec2f image2Point = cam2.GetImagePosition(cam2Dir);

        // Output results
        cout << "Center pixel of first image (" << image1Point << ") "
                << "with depth " << d << " corresponds to (" << image2Point
                << ") on second image." << endl;
    }
}

vector<S> Controller::LoadFilenamesFromFile(const char* filename) {
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

vector<pair<CameraDataf, CameraPoseDataf> > Controller::LoadCamerasFromFile(
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

