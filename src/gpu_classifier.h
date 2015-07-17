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


int train_gpu(int* , float* , int , float *);

int predict_gpu(float* , float* , int , int* );

#endif // GPU_CLASSIFIER_H
