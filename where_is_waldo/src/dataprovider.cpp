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
    // create new rapid json document.
    Document d;
    d.SetObject();

    //d.AddMember("file", t.file.toString().toWCharArray(), d.GetAllocator());
    d.AddMember("orig_img_width", t.orig_img_width, d.GetAllocator());
    d.AddMember("orig_img_height", t.orig_img_height, d.GetAllocator());
    d.AddMember("sub_img_start_x", t.sub_img_start.x(), d.GetAllocator());
    d.AddMember("sub_img_start_y", t.sub_img_start.y(), d.GetAllocator());
    d.AddMember("sub_img_width", t.sub_img_width, d.GetAllocator());
    d.AddMember("sub_img_heigth", t.sub_img_heigth, d.GetAllocator());
    d.AddMember("top_x", t.top.x(), d.GetAllocator());
    d.AddMember("top_y", t.top.y(), d.GetAllocator());
    d.AddMember("bottom_x", t.bottom.x(), d.GetAllocator());
    d.AddMember("bottom_y", t.bottom.y(), d.GetAllocator());
    d.AddMember("area1", "", d.GetAllocator());
    d.AddMember("area2", "", d.GetAllocator());
    d.AddMember("area3", "", d.GetAllocator());

    // start writing data to json file.
    FILE * jsonFile = fopen(REF_JSON_TRAINING, "w");
    char writeBuffer[65536];
    FileWriteStream os(jsonFile, writeBuffer, sizeof(writeBuffer));
    Writer<FileWriteStream> writer(os);
    d.Accept(writer);
    fclose(jsonFile);

    return;
}

void DataProvider::saveMarkedTrainingData(QList<WaldoMarker> waldos)
{
    Document d;
    d.SetArray();

    foreach(const WaldoMarker &w, waldos) {
        Value v;
        v.SetObject();

        //v.AddMember("file", w.file.fileName().toWCharArray(), d.GetAllocator());
        v.AddMember("sub_img_heigth", w.sub_img_heigth, d.GetAllocator());
        v.AddMember("sub_img_width", w.sub_img_width, d.GetAllocator());
        v.AddMember("sub_img_start_x", w.sub_img_start.x(), d.GetAllocator());
        v.AddMember("sub_img_start_y", w.sub_img_start.y(), d.GetAllocator());
    }

    FILE * jsonFile = fopen(REF_JSON_MARKED, "w");
    char writeBuffer[65536];
    FileWriteStream os(jsonFile, writeBuffer, sizeof(writeBuffer));
    Writer<FileWriteStream> writer(os);
    d.Accept(writer);
    fclose(jsonFile);

    return;
}

void DataProvider::saveAllImages(QList<QUrl> images)
{
    Document d;
    d.SetObject();

    int i = 0;
    foreach (const QUrl &image, images) {
        //d.AddMember("image_"+i, image.fileName()., d.GetAllocator());
        i++;
    }

    FILE * jsonFile = fopen(REF_JSON_ALL, "w");
    char writeBuffer[65536];
    FileWriteStream os(jsonFile, writeBuffer, sizeof(writeBuffer));
    Writer<FileWriteStream> writer(os);
    d.Accept(writer);
    fclose(jsonFile);

    return;
}

TrainingData DataProvider::loadSelectedWaldo()
{
    FILE * jsonFile = fopen(REF_JSON_TRAINING, "r");
    char readBuffer[65536];
    rapidjson::FileReadStream is(jsonFile, readBuffer, sizeof(readBuffer));
    rapidjson::Document d;
    d.ParseStream<0>(is);

    TrainingData data = TrainingData();
    //data.file = d["file"];
    QUrl url;
    url.setUrl(d["file"].GetString());
    data.file = url;
    data.orig_img_width = d["orig_img_width"].GetInt();
    data.orig_img_height = d["orig_img_height"].GetInt();
    d["sub_img_start"];
    data.sub_img_width = d["sub_img_width"].GetInt();
    data.sub_img_heigth = d["sub_img_heigth"].GetInt();

    QPoint p;
    p.setX(d["top_x"].GetInt());
    p.setY(d["top_y"].GetInt());
    data.top = p;

    QPoint b;
    b.setX(d["bottom_x"].GetInt());
    b.setY(d["bottom_y"].GetInt());
    data.bottom = b;

    d["bottom"];
    d["area1"];
    d["area2"];
    d["area3"];




    return data;
    //TODO
}

QList<WaldoMarker> DataProvider::loadMarkedTrainingData()
{
    return QList<WaldoMarker>();
    //TODO
}

QList<QUrl> DataProvider::loadAllImages()
{
    return QList<QUrl>();
    //TODO
}

