#include "feature.h"

Feature::Feature(TrainingData* tdata)
{
    data = tdata;
}

void Feature::createFeatures()
{
    feature1 = createSingleFeature(data->sub_img_width, data->sub_img_heigth, data->area1);
    feature2 = createSingleFeature(data->sub_img_width, data->sub_img_heigth, data->area2);
    feature3 = createSingleFeature(data->sub_img_width, data->sub_img_heigth, data->area3);
}

float* Feature::createSingleFeature(int width, int heigth, QPainterPath area)
{
    //Allocate Memory width*heigth = num_pixel, *10 because each pixel has 10 features
    float* feature = new float[width * heigth * 10];
    int x=0,y=0,pixel_idx=0,feature_idx=0;
    float* img;

    int numpix = width * heigth;

    ppm::readPPM(REF_IMG, width, heigth, &img);

    for(int i = 0; i < numpix; i++)
    {
        qDebug() << numpix << i;
        x = i % heigth;
        y = (int)(i / heigth);

        //* 3 because each pixel has r,g,b
        pixel_idx = i * 3;

        float r = img[pixel_idx];
        float g = img[pixel_idx+1];
        float b = img[pixel_idx+2];

        //make all permutations for feature generation
        feature_idx = i * 10;
        feature[feature_idx] = r;
        feature[feature_idx+1] = g;
        feature[feature_idx+2] = b;
        feature[feature_idx+3] = r*r;
        feature[feature_idx+4] = g*g;
        feature[feature_idx+5] = b*b;
        feature[feature_idx+6] = r*g;
        feature[feature_idx+7] = r*b;
        feature[feature_idx+8] = g*b;

        //if pixel in area then last feature is 1 else -1 like in paper
        if(area.contains(QPoint(x,y)))
        {
            feature[feature_idx+9] = 1;
        }
        else
        {
            feature[feature_idx+9] = -1;
        }
    }

    return feature;
}

float* Feature::getFeature(int nr){
    switch(nr)
    {
        case 1: return feature1;
        case 2: return feature2;
        case 3: return feature3;
    }
}
