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



//label and features are input variables, beta is the output variable
void train_gpu(int* labels, float* features, int num_features, double *beta)
{
    for(int i = 0; i < EPOCHS; i++)
    {
            double gradient[FEAT_LEN] = {};
            for(int k = 0; k < num_features; k++) {
                int output = labels[k];


                //find z
                double z = 0;
                for(int i = 0; i < FEAT_LEN; i++) {
                        z += beta[i] * features[k*FEAT_LEN+ i];
                }

                //calc sigmoid
                double prob_y = (1 / (1 + exp(-z)));

                for(int j = 0; j < FEAT_LEN; j++) {
                    gradient[j] += (double)features[k*FEAT_LEN+j]*(output - prob_y);
                }

            }

            //qDebug() << ":" << beta[0]<< ":" << beta[1]<< ":" << beta[2]<< ":" << beta[3]<< ":" << beta[4]<< ":" << beta[5]<< ":" << beta[6]<< ":" << beta[7]<< ":" << beta[8] << "---- lik:" << lik;
            //qDebug()<<i;

            //update beta
            for(int i = 0; i < FEAT_LEN; i++) {
                    beta[i] += LEARN_CONST * gradient[i];
            }

            //qDebug() << ":" << beta[0]<< ":" << betas[1]<< ":" << betas[2]<< ":" << betas[3]<< ":" << betas[4]<< ":" << betas[5]<< ":" << betas[6]<< ":" << betas[7]<< ":" << betas[8] << "---- cor:" << betas[9];

   }

}

//features and beta is the input variable, predictions is the output variable
void predict_gpu(float* features, double* beta, int num_features, int* predictions)
{
    for(int i = 0; i < num_features; i++) {
        //find z
        double z = 0;
        for(int j = 0; j < FEAT_LEN; j++) {
                z += beta[j] * features[i*FEAT_LEN+ j];
        }

        //calc sigmoid
        double prob_y = (1 / (1 + exp(-z)));

        int estimated_class;
        if(prob_y > 0.5) estimated_class = 1;
        else estimated_class = 0;

        predictions[i] = estimated_class;
    }
}
