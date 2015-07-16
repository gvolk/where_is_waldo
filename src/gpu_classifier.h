#ifndef GPU_CLASSIFIER_H
#define GPU_CLASSIFIER_H

#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <math.h>

using std::cout;
using std::cerr;
using std::endl;
using std::exception;
using std::string;


void train_gpu(int* labels, float* features, double *beta);

void predict_gpu(float* features, double* beta, int* precitions);

#endif // GPU_CLASSIFIER_H
