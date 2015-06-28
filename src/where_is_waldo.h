#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <math.h>

using std::cout;
using std::cerr;
using std::endl;
using std::exception;
using std::string;


int run(string imagePath, string outputPath);
float *cpuReciprocal(float *data, unsigned size);
float *gpuReciprocal(float *data, unsigned size);
