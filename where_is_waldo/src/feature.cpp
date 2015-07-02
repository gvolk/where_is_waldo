#include "feature.h"

Feature::Feature(TrainingData* tdata)
{
    data = tdata;
}

void Feature::createFeatures()
{
    feature1 = createSingleFeature(data->orig_img_width, data->sub_img_heigth, data->area1);
    feature2 = createSingleFeature(data->orig_img_width, data->sub_img_heigth, data->area2);
    feature3 = createSingleFeature(data->orig_img_width, data->sub_img_heigth, data->area3);
}

float* Feature::createSingleFeature(int width, int heigth, QPainterPath area)
{
    //Allocate Memory width*heigth = num_pixel, *10 because each pixel has 10 features
    float* feature = new float[width * heigth * 10];

    float* img;

    ppm::readPPM(REF_IMG, width, heigth, &img);

    for(int x = 0; x < width; x++)
    {
        for(int y = 0; y < heigth; y++)
        {
            //* 3 because each pixel has r,g,b
            int i = (y*width + x) * 3;

            float r = img[i];
            float g = img[i+1];
            float b = img[i+2];

            //make all permutations for feature generation
            feature[i] = r;
            feature[i+1] = g;
            feature[i+2] = b;
            feature[i+3] = r*r;
            feature[i+4] = g*g;
            feature[i+5] = b*b;
            feature[i+6] = r*g;
            feature[i+7] = r*b;
            feature[i+8] = g*b;

            //if pixel in area then last feature is 1 else -1 like in paper
            if(area.contains(QPoint(x,y)))
            {
                feature[i+9] = 1;
            }
            else
            {
                feature[i+9] = -1;
            }

        }
    }

    return feature;
}
