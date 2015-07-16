#include "feature.h"

Feature::Feature(TrainingData* tdata)
{
    data = tdata;
    featurefile = REF_IMG;
}

Feature::Feature(TrainingData* tdata, const char* file)
{
    data = tdata;
    featurefile  = file;
}

void Feature::createFeatures()
{
    feature1 = createSingleFeature(data->sub_img_width, data->sub_img_heigth, data->area1);
    feature2 = createSingleFeature(data->sub_img_width, data->sub_img_heigth, data->area2);
    feature3 = createSingleFeature(data->sub_img_width, data->sub_img_heigth, data->area3);

    doGauss(REF_AREA, GAUSS_AREA);
    for(int i= 0; i<50; i++)
    {
    doGauss(GAUSS_AREA, GAUSS_AREA);
    }
}

feature_data* Feature::createSingleFeature(int width, int heigth, QPainterPath area)
{
    //Allocate Memory width*heigth = num_pixel, *10 because each pixel has 10 features
    int numpix = width * heigth;
    feature_data* tmpdata = new feature_data();
    tmpdata->features = new float[numpix * FEAT_LEN];
    tmpdata->labels = new int[numpix];
    tmpdata->num_pix_features = numpix;
    int x=0,y=0,pixel_idx=0,feature_idx=0;
    float* img;

    ppm::readPPM(featurefile, width, heigth, &img);

    for(int i = 0; i < numpix; i++)
    {
        x = i % width;
        y = (int)(i / width);

        //* 3 because each pixel has r,g,b
        pixel_idx = i * 3;


        float r = img[pixel_idx];
        float g = img[pixel_idx+1];
        float b = img[pixel_idx+2];

        //make all permutations for feature generation
        feature_idx = i * FEAT_LEN;
        tmpdata->features[feature_idx] = r;
        tmpdata->features[feature_idx+1] = g;
        tmpdata->features[feature_idx+2] = b;
        tmpdata->features[feature_idx+3] = r*r;
        tmpdata->features[feature_idx+4] = g*g;
        tmpdata->features[feature_idx+5] = b*b;
        tmpdata->features[feature_idx+6] = r*g;
        tmpdata->features[feature_idx+7] = r*b;
        tmpdata->features[feature_idx+8] = g*b;

        //if pixel in area then last feature is 1 else 0 like in paper
        if(area.contains(QPoint(x,y)))
        {
            tmpdata->labels[i] = 1;
        }
        else
        {
            tmpdata->labels[i] = 0;
        }
    }

    for(int i=0; i<FEAT_LEN; i++)
    {
        normalizeFeature(i,tmpdata);
    }

    return tmpdata;
}

void Feature::normalizeFeature(int feature_idx, feature_data* features)
{
    float sum, min=10000, max=0, mean;
    float feature;
    int idx;
    for(int i=0; i< features->num_pix_features; i++)
    {
        idx = i*FEAT_LEN + feature_idx;
        feature = features->features[idx];
        if(feature < min)
        {
            min = feature;
        }
        if(feature > max)
        {
            max = feature;
        }
        sum += feature;
    }
    mean = sum / features->num_pix_features;

    for(int i=0; i< features->num_pix_features; i++)
    {
        feature = features->features[idx];
        features->features[idx] = (feature - mean)/(max-min);
    }
}


feature_data* Feature::getFeature(int nr){
    switch(nr)
    {
        case 1: return feature1;
        case 2: return feature2;
        case 3: return feature3;
        default: return feature1;
    }
}
