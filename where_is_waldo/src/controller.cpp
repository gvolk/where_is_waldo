#include "controller.h"

#define NOISE_FACTOR 2
#define TMP_IMG "tmp_output.ppm"

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

void Controller::compareGPUvsCPU(LogRegClassifier* c1,LogRegClassifier* c2,LogRegClassifier* c3, Feature* f )
{
    std::clock_t startTime;
    std::clock_t trainingTime;
    std::clock_t predictionTime;
    //run cpu
    c1->set_mode(CPU_MODE);
    c2->set_mode(CPU_MODE);
    c3->set_mode(CPU_MODE);

    startTime =clock();

    c1->train(f->getFeature(1));
    c2->train(f->getFeature(2));
    c3->train(f->getFeature(3));

    trainingTime = clock();

    c1->test_classification(f->getFeature(1), f->getFeature(1));
    c2->test_classification(f->getFeature(2), f->getFeature(2));
    c3->test_classification(f->getFeature(3), f->getFeature(3));

    predictionTime = clock();

    double cpu_train = double(trainingTime - startTime) / CLOCKS_PER_SEC;
    double cpu_pred = double(predictionTime - trainingTime) / CLOCKS_PER_SEC;


    //run gpu
    c1->set_mode(GPU_MODE);
    c2->set_mode(GPU_MODE);
    c3->set_mode(GPU_MODE);

    startTime =clock();

    c1->train(f->getFeature(1));
    c2->train(f->getFeature(2));
    c3->train(f->getFeature(3));

    trainingTime = clock();

    c1->test_classification(f->getFeature(1), f->getFeature(1));
    c2->test_classification(f->getFeature(2), f->getFeature(2));
    c3->test_classification(f->getFeature(3), f->getFeature(3));

    predictionTime = clock();

    double gpu_train = double(trainingTime - startTime) / CLOCKS_PER_SEC;
    double gpu_pred = double(predictionTime - trainingTime) / CLOCKS_PER_SEC;

    qDebug() << "cpu train:" << cpu_train << "cpu predict" << cpu_pred;
    qDebug() << "gpu train:" << gpu_train << "gpu predict" << gpu_pred;
}

void Controller::search_waldo(QList<QUrl> urls, TrainingData *data)
{
    f = new Feature(data);
    f->createFeatures();
    c1_class = new LogRegClassifier(GPU_MODE);
    c2_class = new LogRegClassifier(GPU_MODE);
    c3_class = new LogRegClassifier(GPU_MODE);

    for(int i=0; i<9; i++)
    {
        qDebug() << f->getFeature(1)->features[i];
    }
    qDebug() << f->getFeature(1)->labels[0];


    Feature* f2 = new Feature(data, "training_image_2.ppm");
    f2->createFeatures();

    compareGPUvsCPU(c1_class, c2_class, c3_class, f);

    /*
    c1_class->train(f->getFeature(1));
    c1_class->test_classification(f->getFeature(1), f->getFeature(1));

    c2_class->train(f->getFeature(2));
    c2_class->test_classification(f->getFeature(2), f->getFeature(2));

    c3_class->train(f->getFeature(3));
    c3_class->test_classification(f->getFeature(3), f->getFeature(3));
        */

    // do for all url to compare.
    foreach(const QUrl url, urls) {
        if (data->file == url) {
            // skip ref file.
            continue;
        }

        // create S s1 and set s1.path = filename. (some for s2).
        vector<QPoint> topPoints;
        vector<QPoint> bottomPoints;
        vector<QPoint> startPoints;

        S s1;
        s1.path = data->file.toString().toStdString().c_str();
        S s2;
        s2.path = url.toString().toStdString().c_str();

        topPoints = GetRefPoints(s1, s2, data->top);
        bottomPoints = GetRefPoints(s1, s2, data->bottom);
        startPoints = GetRefPoints(s1, s2, data->sub_img_start);

        QPixmap tmpImg(url.fileName());

        QPoint minTop(tmpImg.width(), tmpImg.height());
        QPoint maxBottom(0, 0);
        QPoint minStart(tmpImg.width(), tmpImg.height());

        for (unsigned int i = 0; i < topPoints.size(); i++) {
            QPoint top = topPoints[i];
            QPoint bottom = bottomPoints[i];
            QPoint start = startPoints[i];

            //float diff = GetDiffFactor(data->top, data->bottom, top, bottom);

            if (top.y() < minTop.y()) {
                minTop.setY(top.y());
            }
            if (bottom.y() < maxBottom.y()) {
                maxBottom.setY(bottom.y());
            }
            if (start.x() < minStart.x()) {
                minStart.setX(start.x());
            }
            if (start.y() < minStart.y()) {
                minStart.setY(start.y());
            }
        }

        // we have now a minimum top.y and a bottom.y
        float diffY = maxBottom.y() - minTop.y();

        // now take a factor! sensible do not choose big.
        diffY = diffY * NOISE_FACTOR;

        float tmpDiffY = data->bottom.y() - data->top.y();
        float tmpDiffX = data->bottom.x() - data->top.x();

        float tmpQuotient = diffY / tmpDiffY;

        float diffX = tmpDiffX * tmpQuotient;

        //save original heigth and width of image

        QRect rect(minStart.x(), minStart.y(), diffX, diffY);
        QPixmap cropped = tmpImg.copy(rect);

        QFile file(TMP_IMG);
        file.open(QIODevice::WriteOnly);
        cropped.save(&file, "PPM");

        // @TODO go one here
    }
}

/*float Controller::GetDiffFactor(QPoint top1, QPoint bottom1, QPoint top2, QPoint bottom2) {
    float dist1 = bottom1.y() - top1.y();
    float dist2 = bottom2.y() - top2.y();

    return dist2/dist1;
}*/

vector<QPoint> Controller::GetRefPoints(S s1, S s2, QPoint point) {
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
    vector<QPoint> result;

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
        return vector<QPoint>();
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
    Vec2f image1Point(point.x(), point.y());

    // Process several depths
    for (float d = 0.4f; d <= 1.6f; d += 0.3f)
    {
        //### DO TOP POINT
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

        QPoint point2;
        point2.setX(image2Point[0]);
        point2.setY(image2Point[1]);


        result.push_back(point2);
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

        char path[512];

        // only path is relevant.
        float tmp;
        sscanf(line.c_str(), "%s	%f %f %f %f %f %f %f %f", path, &tmp,
                        &tmp, &tmp, &tmp, &tmp, &tmp, &tmp, &tmp);
        // now save path

        s.path = path;

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

