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
    c1_class = new LogRegClassifier(CPU_MODE);
    c2_class = new LogRegClassifier(CPU_MODE);
    c3_class = new LogRegClassifier(CPU_MODE);

    for(int i=0; i<9; i++)
    {
        qDebug() << f->getFeature(1)->features[i];
    }
    qDebug() << f->getFeature(1)->labels[0];


    Feature* f2 = new Feature(data, "training_image_2.ppm");
    f2->createFeatures();

    c1_class->train(f->getFeature(1));
    c1_class->test_classification(f->getFeature(1), f->getFeature(1));

    c2_class->train(f->getFeature(2));
    c2_class->test_classification(f->getFeature(2), f->getFeature(2));

    c3_class->train(f->getFeature(3));
    c3_class->test_classification(f->getFeature(3), f->getFeature(3));

    /*
    c1_class->train(f->getFeature(1));
    c1_class->test_classification(f->getFeature(1), f->getFeature(1));

    c2_class->train(f->getFeature(2));
    c2_class->test_classification(f->getFeature(2), f->getFeature(2));

    c3_class->train(f->getFeature(3));
    c3_class->test_classification(f->getFeature(3), f->getFeature(3));
        */
    //TODO
    // take new method
    // create S s1 and set s1.path = filename. (some for s2).
    vector<pair<QPoint, QPoint> > tmp;
    S s1;
    s1.path = "training_image.ppm";
    S s2;
    s2.path = "training_image_2.ppm";
    tmp = GetRefPoints(s1, s2, data->top, data->bottom);

    for (int i = 0; i < tmp.size(); i++) {
        QPoint top = tmp[i].first;
        QPoint bottom = tmp[i].second;

        float diff = GetDiffFactor(data->top, data->bottom, top, bottom);

        // now
        // diff = diff * 2; // i.e. to increase test bild size.
    }
}

float Controller::GetDiffFactor(QPoint top1, QPoint bottom1, QPoint top2, QPoint bottom2) {
    float dist1 = bottom1.y() - top1.y();
    float dist2 = bottom2.y() - top2.y();

    return dist2/dist1;
}

vector<pair<QPoint, QPoint> > Controller::GetRefPoints(S s1, S s2, QPoint top, QPoint bottom) {
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
    vector<pair<QPoint, QPoint> > result;

    string tmp = "camera_loading/test.nvm";
    const char* filename = tmp.c_str();

    // Load file paths.
    vector<S> paths = LoadFilenamesFromFile(filename);
    signed int index1 = -1;
    signed int index2 = -1;
    for (unsigned int i = 0; i < paths.size(); i++) {
        if (paths[i].path == s1.path) {
            index1 = i;
        } else if (paths[i].path == s2.path) {
            index2 = i;
        }

        if (index1 != -1 && index2 != -1) {
            break;
        }
    }

    if (index1 == -1 || index2 == -1) {
        return vector<pair<Vec2f, Vec2f> >();
    }

    // Load camera data.
    vector<pair<CameraDataf, CameraPoseDataf> > cameraData =
                LoadCamerasFromFile(filename);

    // Pick to cameras
    CameraDataf cam1 = cameraData[index1].first;
    CameraPoseDataf pose1 = cameraData[index1].second;
    CameraDataf cam2 = cameraData[index2].first;
    CameraPoseDataf pose2 = cameraData[index2].second;

    // === Project a point from one camera to the other ===
    //Vec2f image1Point(cam1.ImageSize / 2); // Pick a point at the center of the image
    Vec2f image1PointTop(top.x(), top.y());
    Vec2f image1PointBottom(bottom.x(), bottom.y());

    // Process several depths
    for (float d = 0.4f; d <= 1.6f; d += 0.3f)
    {
        // Transform point to viewspace vector
        Vec3f cam1Dir = cam1.GetViewspaceDirection(image1PointTop);
        // Transform to worldspace ray
        Rayf worldRay = pose1.GetWorldspaceRay(cam1Dir);

        // Pick a point on the ray according to depth
        Vec3f worldPoint = worldRay.Origin + worldRay.Direction * d;
        // ATTENTION: The depth is interpreted along the normalized ray direction, not along the principal camera axis!
        // Maybe you have to change this.

        // Transform to second camera's viewspace
        Vec3f cam2Dir = pose2.GetViewspaceDirectionFromPoint(worldPoint);
        // Transform to second image
        Vec2f image2PointTop = cam2.GetImagePosition(cam2Dir);

        // Output results
        cout << "Center pixel of first image (" << image1PointTop << ") "
                << "with depth " << d << " corresponds to (" << image2PointTop
                << ") on second image." << endl;

        // Transform point to viewspace vector
        cam1Dir = cam1.GetViewspaceDirection(image1PointBottom);
        // Transform to worldspace ray
        worldRay = pose1.GetWorldspaceRay(cam1Dir);

        // Pick a point on the ray according to depth
        worldPoint = worldRay.Origin + worldRay.Direction * d;
        // ATTENTION: The depth is interpreted along the normalized ray direction, not along the principal camera axis!
        // Maybe you have to change this.

        // Transform to second camera's viewspace
        cam2Dir = pose2.GetViewspaceDirectionFromPoint(worldPoint);
        // Transform to second image
        Vec2f image2PointBottom = cam2.GetImagePosition(cam2Dir);

        // Output results
        cout << "Center pixel of first image (" << image1PointBottom << ") "
                << "with depth " << d << " corresponds to (" << image2PointBottom
                << ") on second image." << endl;

        QPoint top2;
        top2.setX(image2PointTop[0]);
        top2.setY(image2PointTop[1]);

        QPoint bottom2;
        bottom2.setX(image2PointBottom[0]);
        bottom2.setY(image2PointBottom[1]);
        result.push_back(make_pair(top2, bottom2));
    }

    return result;
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
    gc->displayImage();
}

