#ifndef DATAPROVIDER_H
#define DATAPROVIDER_H

#include "data.h"

class DataProvider
{
public:
    DataProvider();
    void saveSelectedWaldo(TrainingData);
    void saveMarkedTrainingData(WaldoMarker);
    void saveAllImages(QList<QUrl> images);
    TrainingData loadSelectedWaldo();
    QList<WaldoMarker> loadMarkedTrainingData();
    QList<QUrl> loadAllImages();
};

#endif // DATAPROVIDER_H
