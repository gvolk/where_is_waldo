#ifndef FEATURE_H
#define FEATURE_H

#include "data.h"
#include "../src/PPM.hh"

class Feature
{
public:
    Feature(TrainingData*);
    void createFeatures();
    float* getFeature(int nr);

private:
    float* createSingleFeature(int width, int height, QPainterPath area);
    TrainingData* data;
    float* feature1;
    float* feature2;
    float* feature3;


};

#endif // FEATURE_H
