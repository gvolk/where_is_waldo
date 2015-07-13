#include "dataprovider.h"

using namespace rapidjson;
using namespace std;

#define REF_JSON_TRAINING "training_data.json"
#define REF_JSON_ALL "all_images.json"
#define REF_JSON_MARKED "marked_training_data.json"

DataProvider::DataProvider()
{
}

void DataProvider::saveSelectedWaldo(TrainingData t)
{
    FILE * jsonFile = fopen(REF_JSON_TRAINING, "w");
    char writeBuffer[65536];
    FileWriteStream os(jsonFile, writeBuffer, sizeof(writeBuffer));
    Writer<FileWriteStream> writer(os);

    writer.StartObject();

    writer.String("file");
    writer.String(t.file.toString().toStdString().c_str());

    writer.String("orig_img_width");
    writer.Int(t.orig_img_width);

    writer.String("orig_img_height");
    writer.Int(t.orig_img_height);

    writer.String("sub_img_start_x");
    writer.Int(t.sub_img_start.x());

    writer.String("sub_img_start_y");
    writer.Int(t.sub_img_start.y());

    writer.String("sub_img_width");
    writer.Int(t.sub_img_width);

    writer.String("sub_img_height");
    writer.Int(t.sub_img_heigth);

    writer.String("top_x");
    writer.Int(t.top.x());

    writer.String("top_y");
    writer.Int(t.top.y());

    writer.String("bottom_x");
    writer.Int(t.bottom.x());

    writer.String("bottom_y");
    writer.Int(t.bottom.y());

    writer.EndObject();

    fclose(jsonFile);

    QFile file1("area1.dat");
    file1.open(QIODevice::WriteOnly);
    QDataStream out1(&file1);
    out1 << t.area1;
    file1.close();

    QFile file2("area2.dat");
    file2.open(QIODevice::WriteOnly);
    QDataStream out2(&file2);
    out2 << t.area2;
    file2.close();

    QFile file3("area3.dat");
    file3.open(QIODevice::WriteOnly);
    QDataStream out3(&file3);
    out3 << t.area3;
    file3.close();

    return;
}

void DataProvider::saveMarkedTrainingData(QList<WaldoMarker> waldos)
{
    FILE * jsonFile = fopen(REF_JSON_MARKED, "w");
    char writeBuffer[65536];
    FileWriteStream os(jsonFile, writeBuffer, sizeof(writeBuffer));
    Writer<FileWriteStream> writer(os);

    writer.StartObject();
    writer.String("waldos");
    writer.StartArray();
    foreach(const WaldoMarker &w, waldos) {
        writer.StartObject();

        writer.String("file");
        writer.String(w.file.toString().toStdString().c_str());

        writer.String("sub_img_height");
        writer.Int(w.sub_img_heigth);

        writer.String("sub_img_width");
        writer.Int(w.sub_img_width);

        writer.String("sub_img_start_x");
        writer.Int(w.sub_img_start.x());

        writer.String("sub_img_start_y");
        writer.Int(w.sub_img_start.y());

        writer.EndObject();
    }
    writer.EndArray();
    writer.EndObject();

    fclose(jsonFile);

    return;
}

void DataProvider::saveAllImages(QList<QUrl> images)
{
    FILE * jsonFile = fopen(REF_JSON_ALL, "w");
    char writeBuffer[65536];
    FileWriteStream os(jsonFile, writeBuffer, sizeof(writeBuffer));
    Writer<FileWriteStream> writer(os);

    writer.StartObject();
    writer.String("images");
    writer.StartArray();
    foreach (const QUrl &image, images) {
        writer.String(image.toString().toStdString().c_str());
    }
    writer.EndArray();
    writer.EndObject();

    fclose(jsonFile);

    return;
}

TrainingData DataProvider::loadSelectedWaldo()
{
    FILE * jsonFile = fopen(REF_JSON_TRAINING, "r");
    char readBuffer[65536];
    FileReadStream is(jsonFile, readBuffer, sizeof(readBuffer));
    Document d;
    d.ParseStream(is);

    fclose(jsonFile);

    if (!d.IsObject()) {
        qDebug() << "Error: loadingSelectedWaldo(): d is no object";
        return TrainingData();
    }

    TrainingData td;

    QUrl file;
    file.setUrl(d["file"].GetString());
    td.file = file;

    td.orig_img_height = d["orig_img_height"].GetInt();
    td.orig_img_width = d["orig_img_width"].GetInt();

    QPoint sub_img_start;
    sub_img_start.setX(d["sub_img_start_x"].GetInt());
    sub_img_start.setY(d["sub_img_start_y"].GetInt());
    td.sub_img_start = sub_img_start;

    td.sub_img_heigth = d["sub_img_height"].GetInt();
    td.sub_img_width = d["sub_img_width"].GetInt();

    QPoint top;
    top.setX(d["top_x"].GetInt());
    top.setY(d["top_y"].GetInt());
    td.top = top;

    QPoint bottom;
    bottom.setX(d["bottom_x"].GetInt());
    bottom.setY(d["bottom_y"].GetInt());
    td.bottom = bottom;

    QFile file1("area1.dat");
    file1.open(QIODevice::ReadOnly);
    QDataStream in1(&file1);
    QPainterPath path1;
    in1 >> path1;
    td.area1 = path1;
    file1.close();

    QFile file2("area2.dat");
    file2.open(QIODevice::ReadOnly);
    QDataStream in2(&file2);
    QPainterPath path2;
    in2 >> path2;
    td.area2 = path2;
    file2.close();

    QFile file3("area3.dat");
    file3.open(QIODevice::ReadOnly);
    QDataStream in3(&file3);
    QPainterPath path3;
    in3 >> path3;
    td.area3 = path3;
    file3.close();

    return td;
}

QList<WaldoMarker> DataProvider::loadMarkedTrainingData()
{
    FILE * jsonFile = fopen(REF_JSON_MARKED, "r");
    char readBuffer[65536];
    FileReadStream is(jsonFile, readBuffer, sizeof(readBuffer));
    Document d;
    d.ParseStream(is);

    fclose(jsonFile);

    if (!d.IsObject()) {
        qDebug() << "Error: loadMarkedTrainingData(): d is no object";
        return QList<WaldoMarker>();
    }

    const Value& w = d["waldos"];

    if (w.IsArray()) {
        QList<WaldoMarker> waldos;
        for (SizeType i = 0; i < w.Size(); i++) {
            WaldoMarker wm;
            QUrl file;
            file.setUrl(w[i]["file"].GetString());
            wm.file = file;
            wm.sub_img_heigth = w[i]["sub_img_height"].GetInt();
            wm.sub_img_width = w[i]["sub_img_width"].GetInt();
            QPoint point;
            point.setX(w[i]["sub_img_start_x"].GetInt());
            point.setY(w[i]["sub_img_start_y"].GetInt());
            wm.sub_img_start = point;
            waldos.append(wm);
        }
        return waldos;
    } else {
        return QList<WaldoMarker>();
    }
}

QList<QUrl> DataProvider::loadAllImages()
{
    FILE * jsonFile = fopen(REF_JSON_ALL, "r");
    char readBuffer[65536];
    FileReadStream is(jsonFile, readBuffer, sizeof(readBuffer));
    Document d;
    d.ParseStream(is);

    fclose(jsonFile);

    if (!d.IsObject()) {
        qDebug() << "Error: loadAllImages(): d is no object";
        return QList<QUrl>();
    }

    const Value& img = d["images"];

    if (img.IsArray()) {
        QList<QUrl> urls;
        for (SizeType i = 0; i < img.Size(); i++) {
            QUrl url;
            url.setUrl(img[i].GetString());
            urls.append(url);
        }
        return urls;
    } else {
        return QList<QUrl>();
    }
}
