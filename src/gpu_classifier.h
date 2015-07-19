#ifndef GPU_CLASSIFIER_H
#define GPU_CLASSIFIER_H

#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <math.h>

#include "../where_is_waldo/src/defines.h"


using std::cout;
using std::cerr;
using std::endl;
using std::exception;
using std::string;


int train_gpu(int* , float* , int , double *);

int predict_gpu(float* , double* , int , int* );

//std::pair<float,float> calcPCorrect(feature_data* test, feature_data* train_data);
std::pair<float,float> calc_P_Correct(int* labels, int* predicted, int num_features);

#endif // GPU_CLASSIFIER_H
