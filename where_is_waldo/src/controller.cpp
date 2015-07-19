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

bool checkWaldo(const char* imagepath)
{

}

void Controller::testClassifier(TrainingData *data)
{
    data->sub_img_start = QPoint(0,0);
    data->sub_img_heigth = data->orig_img_height;
    data->sub_img_width = data->orig_img_width;

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

    c1_class->train(f->getFeature(1));
    c1_class->test_classification(f->getFeature(1), f->getFeature(1));

    c2_class->train(f->getFeature(2));
    c2_class->test_classification(f->getFeature(2), f->getFeature(2));

    c3_class->train(f->getFeature(3));
    c3_class->test_classification(f->getFeature(3), f->getFeature(3));

}

void Controller::search_waldo(QList<QUrl> urls, TrainingData *data)
{
    testClassifier(data);
    return;
    /*f = new Feature(data);
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

    compareGPUvsCPU(c1_class, c2_class, c3_class, f);*/

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
        vector<Vec2f> topPoints;
        vector<Vec2f> bottomPoints;
        vector<Vec2f> startPoints;

        string pathStr1 = data->file.toString().toStdString();
        char *path1 = new char[pathStr1.length() + 1];
        strcpy(path1, pathStr1.c_str());

        string pathStr2 = url.toString().toStdString();
        char *path2 = new char[pathStr2.length() + 1];
        strcpy(path2, pathStr2.c_str());

        QPoint tmpTop;
        tmpTop.setX(data->top.x() + data->sub_img_start.x());
        tmpTop.setY(data->top.y() + data->sub_img_start.y());

        QPoint tmpBottom;
        tmpBottom.setX(data->bottom.x() + data->sub_img_start.x());
        tmpBottom.setY(data->bottom.y() + data->sub_img_start.y());

        startPoints = GetRefPoints(path1, path2, data->sub_img_start);
        topPoints = GetRefPoints(path1, path2, tmpTop);
        bottomPoints = GetRefPoints(path1, path2, tmpBottom);

        QPixmap tmpImg(path2);

        Vec2f topAverage;
        topAverage[0] = 0.f;
        topAverage[1] = 0.f;

        Vec2f bottomAverage;
        bottomAverage[0] = 0.f;
        bottomAverage[1] = 0.f;

        Vec2f startAverage;
        startAverage[0] = 0.f;
        startAverage[1] = 0.f;

        for (unsigned int i = 0; i < topPoints.size(); i++) {
            Vec2f top = topPoints[i];
            Vec2f bottom = bottomPoints[i];
            Vec2f start = startPoints[i];

            topAverage[0] += top[0];
            topAverage[1] += top[1];

            bottomAverage[0] += bottom[0];
            bottomAverage[1] += bottom[1];

            startAverage[0] += start[0];
            startAverage[1] += start[1];
        }

        topAverage[0] /= topPoints.size();
        topAverage[1] /= topPoints.size();

        bottomAverage[0] /= topPoints.size();
        bottomAverage[1] /= topPoints.size();

        startAverage[0] /= topPoints.size();
        startAverage[1] /= topPoints.size();

        float diffY = bottomAverage[1] - topAverage[1];
        qDebug() << diffY;


        int tmpDiffY = data->sub_img_heigth;
        int tmpDiffX = data->sub_img_width;

        float scaleFactor = diffY / tmpDiffY;

        int diffX = (int)(tmpDiffX * scaleFactor);

        //save original heigth and width of image

        QRect rect(startAverage[0], startAverage[1], diffX, (int)diffY);
        QPixmap cropped = tmpImg.copy(rect);

        qDebug() << (cropped.height() * scaleFactor);
        qDebug() << (cropped.width() * scaleFactor);
        cropped.scaledToHeight(cropped.height() * scaleFactor);
        cropped.scaledToWidth(cropped.width() * scaleFactor);

        QFile file(TMP_IMG);
        file.open(QIODevice::WriteOnly);
        cropped.save(&file, "PPM");

        delete [] path1;
        delete [] path2;

        // @TODO go one here
    }
}

/*float Controller::GetDiffFactor(QPoint top1, QPoint bottom1, QPoint top2, QPoint bottom2) {
    float dist1 = bottom1.y() - top1.y();
    float dist2 = bottom2.y() - top2.y();

    return dist2/dist1;
}*/

vector<Vec2f> Controller::GetRefPoints(const char* s1, const char* s2, QPoint point) {
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
    vector<Vec2f> result;

    string tmp = "../images/cmvs/nvm.cmvs.nvm";
    const char* filename = tmp.c_str();

    // Load file paths.
    vector<S> paths = LoadFilenamesFromFile(filename);
    signed int index1 = -1;
    signed int index2 = -1;
    for (unsigned int i = 0; i < paths.size(); i++) {
        if (ComparePath(paths[i].path, s1)) {
            index1 = i;
        } else if (ComparePath(paths[i].path, s2)) {
            index2 = i;
        }

        if (index1 != -1 && index2 != -1) {
            break;
        }
    }

    if (index1 == -1 || index2 == -1) {
        return vector<Vec2f>();
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
    Vec2f image1Point;

    image1Point[0] = point.x();
    image1Point[1] = point.y();

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

        /*QPoint point2;
        point2.setX(image2Point[0]);
        point2.setY(image2Point[1]);*/


        result.push_back(image2Point);
    }

    return result;
}

bool Controller::ComparePath(const char* path1, const char* path2) {
    QString s1(path1);
    QString s2(path2);

    s1 = s1.replace("/images", "");
    s1 = s1.replace("\"", "");
    s1 = s1.replace(" ", "");

    s2 = s2.replace("/images", "");
    s2 = s2.replace("\"", "");
    s2 = s2.replace(" ", "");

    return (s1 == s2);
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

        s.path = (char*) malloc(512 * sizeof(char));

        char* path = (char*) malloc(512 * sizeof(char));

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

        intrinsics.ImageSize = Vec2f(2848, 2136); // TODO: Set this correctly from image
        //intrinsics.ImageSize = Vec2f(q.width(), q.height());

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

