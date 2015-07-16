#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <math.h>

using std::cout;
using std::cerr;
using std::endl;
using std::exception;
using std::string;


int doGauss(const char* imagePath, const char* outputPath);
float *cpuReciprocal(float *data, unsigned size);
float *gpuReciprocal(float *data, unsigned size);
