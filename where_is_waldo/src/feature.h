#ifndef FEATURE_H
#define FEATURE_H

#include "data.h"
#include "../src/PPM.hh"
#include "../src/where_is_waldo.h"

#define FEAT_LEN 9

struct feature_data{
    int num_pix_features;
    float* features;
    int* labels;
};

class Feature
{
public:
    Feature(TrainingData*);
    Feature(TrainingData*, const char*);
    void createFeatures();
    feature_data* getFeature(int nr);

private:
    const char* featurefile;
    feature_data* createSingleFeature(int width, int height, QPainterPath area);
    TrainingData* data;
    feature_data* feature1;
    feature_data* feature2;
    feature_data* feature3;
};

#endif // FEATURE_H
