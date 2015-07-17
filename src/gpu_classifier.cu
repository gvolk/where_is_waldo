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



#define THREADS_PER_BLOCK 128

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)


__global__ void trainKernel(int* labels, float* features, int num_features, float *beta)
{
    /*int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y;

    if (x >= FEAT_LEN || y >= num_features) {
        return;
    }
    int pos = x + FEAT_LEN * y;

    __shared__ float z;
    __shared__ float proby;

    __shared__ float update[FEAT_LEN];

    for (int epoch = 0; epicg < EPOCHS; epoch++) {
        z  = z + beta[x] * features[pos];

        __syncthreads();

        if(x == 0)
            proby = (1 / (1 + exp(-z)));

        __syncthreads();

        update[x] =
    }

    */

}

//label and features are input variables, beta is the output variable
int train_gpu(int* labels, float* features, int num_features, float *beta)
{
   /* float* gpufeatures;
    int* gpulabels;
    double* gpubeta;

    cudaMalloc((void**) &gpufeatures, num_features * FEAT_LEN * sizeof(float));
    cudaMalloc((void**) &gpulabels, FEAT_LEN * sizeof(int));
    cudaMalloc((void**) &gpubeta, FEAT_LEN * sizeof(double));

    cudaMemcpy(gpufeatures, features, num_features * FEAT_LEN * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpulabels, labels, FEAT_LEN * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemset(gpubeta, 0, FEAT_LEN * sizeof(double));

    // BEGIN run kernel.
    dim3 threadBlock(MAX_THREADS);
    dim3 blockGrid(FEAT_LEN / MAX_THREADS + 1, num_features, 1);

    // run gaussian blur kernel.
    //std::cout<<"would execute kernerl. But skipping an using cpu part at the moment."<<std::endl;
    trainKernel<<< blockGrid, threadBlock >>>(gpulabels, gpufeatures, num_features, gpubeta);

    */

    /*
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

   }*/
    return 0;
}

__global__ void predictKernel(float* features, float* beta, int num_features, int* predictions)
{
    int x = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
    float z, proby;
    int pos;

    if (x  >= num_features) {
        return;
    }

    pos = x * FEAT_LEN ;

    for(int j = 0; j < FEAT_LEN; j++) {
            z += beta[j] * features[pos + j];
    }

    proby = (1.0f / (1.0f + expf(-z)));

    if(proby > 0.5)
    {
        predictions[x] = 1;
    }
    else
    {
        predictions[x] = 0;
    }

}

//features, beta and num_features are the input variables, predictions is the output variable
int predict_gpu(float* features, float* beta, int num_features, int* predictions)
{
    float* gpufeatures;
    float* gpubeta;
    int* gpupredictions;

    std::cout << "starting gpu prediction\n";

    for(int i = 0; i< FEAT_LEN; i++)
    {
        std::cout << beta[i] << endl;
    }

    cudaMalloc((void**) &gpufeatures, num_features * FEAT_LEN * sizeof(float));
    cudaMemcpy(gpufeatures, features, num_features * FEAT_LEN * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void**) &gpubeta, FEAT_LEN * sizeof(float));
    cudaMemcpy(gpubeta, beta,     FEAT_LEN * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void**) &gpupredictions, num_features * sizeof(int));
    cudaMemset(gpupredictions, 0,        num_features * sizeof(int));

    // BEGIN run kernel.
    unsigned int numBlocks = num_features / THREADS_PER_BLOCK ;


    predictKernel<<< numBlocks, THREADS_PER_BLOCK >>>(gpufeatures, gpubeta, num_features, gpupredictions);

    cudaThreadSynchronize();

    cudaMemcpy(predictions, gpupredictions, num_features * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(gpufeatures);
    cudaFree(gpubeta);
    cudaFree(gpupredictions);

    return 0;

}
