/*
 ============================================================================
 Name        : gpu_classifier.cu
 Author      : volk, hettich
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA logistic regression classifier
 ============================================================================
 */
#include "gpu_classifier.h"


#define MAX_THREADS 512

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

# define M_PI           3.14159265358979323846  /* pi */

/**
 * CUDA kernel that computes reciprocal values for a given vector
 */





void train_gpu(int* labels, float* features, double *beta)
{


}

void predict_gpu(float* features, double* beta, int* precitions)
{

}
