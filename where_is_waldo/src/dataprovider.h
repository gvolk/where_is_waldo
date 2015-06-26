#ifndef DATAPROVIDER_H
#define DATAPROVIDER_H

#include "data.h"
#include "rapidjson/document.h"
#include "rapidjson/filewritestream.h"
#include "rapidjson/writer.h"

class DataProvider
{
public:
    DataProvider();
    void saveSelectedWaldo(TrainingData);
    void saveMarkedTrainingData(QList<WaldoMarker>);
    void saveAllImages(QList<QUrl> images);
    TrainingData loadSelectedWaldo();
    QList<WaldoMarker> loadMarkedTrainingData();
    QList<QUrl> loadAllImages();
};

#endif // DATAPROVIDER_H
