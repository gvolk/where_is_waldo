#include "dataprovider.h"

#define REF_JSON "training_data.json"

DataProvider::DataProvider()
{
}

void DataProvider::saveSelectedWaldo(TrainingData t)
{
    // create new rapid json document.
    rapidjson::Document d;
    d.SetObject();

    d.AddMember("file", "", d.GetAllocator());
    d.AddMember("orig_img_width", t.orig_img_width, d.GetAllocator());
    d.AddMember("orig_img_height", t.orig_img_height, d.GetAllocator());
    d.AddMember("sub_img_start", "", d.GetAllocator());
    d.AddMember("sub_img_width", t.sub_img_width, d.GetAllocator());
    d.AddMember("sub_img_heigth", t.sub_img_heigth, d.GetAllocator());
    d.AddMember("top", "", d.GetAllocator());
    d.AddMember("bottom", "", d.GetAllocator());
    d.AddMember("area1", "", d.GetAllocator());
    d.AddMember("area2", "", d.GetAllocator());
    d.AddMember("area3", "", d.GetAllocator());

    // start writing data to json file.
    qDebug() << QDir::currentPath();
    FILE * jsonFile = fopen(REF_JSON, "w");
    char writeBuffer[65536];
    rapidjson::FileWriteStream os(jsonFile, writeBuffer, sizeof(writeBuffer));
    rapidjson::Writer<rapidjson::FileWriteStream> writer(os);
    d.Accept(writer);
    fclose(jsonFile);

    return;
}

void DataProvider::saveMarkedTrainingData(QList<WaldoMarker> w)
{
    return;
    //TODO
}

void DataProvider::saveAllImages(QList<QUrl> images)
{
    return;
    //TODO
}

TrainingData DataProvider::loadSelectedWaldo()
{
    FILE * jsonFile = fopen(REF_JSON, "r");
    char readBuffer[65536];
    rapidjson::FileReadStream is(jsonFile, readBuffer, sizeof(readBuffer));
    rapidjson::Document d;
    d.ParseStream<0>(is);

    TrainingData data = new TrainingData();
    data.file = d["file"];
    data.orig_img_width = d["orig_img_width"];
    data.orig_img_height = d["orig_img_height"];
    d["sub_img_start"];
    data.sub_img_width = d["sub_img_width"];
    data.sub_img_heigth = d["sub_img_heigth"];
    d["top"];
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

