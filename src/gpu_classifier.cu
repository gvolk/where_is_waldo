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


__global__ void trainKernel(int* labels, float* features, int num_features, double *beta, double* gputmpbeta)
{
    int x = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
    double z, proby;
    int pos;

    if (x  >= num_features) {
        return;
    }

    pos = x * FEAT_LEN ;

    for(int j = 0; j < FEAT_LEN; j++) {
            z += beta[j] * (double)features[pos + j];
    }

    proby = (1.0 / (1.0 + exp(-z)));

    for(int j = 0; j < FEAT_LEN; j++) {
        //save #num_features betas of the first feature then #num_features beats for the second for later reduce them
        gputmpbeta[j * num_features + x] += (double) LEARN_CONST * ((double)features[pos + j]*(labels[x] - proby));
    }

}

//reduce betas of one feature previously computed by trainKernel
__global__ void reduceBetas( int num_features, int feature_beta_idx, double* gpuResultBeta1, double* gpuResultBeta2)
{
    int x = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;

    if (x  >= num_features) {
        return;
    }

    extern __shared__ float partialSum[];

    int beta_idx = feature_beta_idx * num_features + x;

    unsigned int t = threadIdx.x;

    partialSum[t] = gpuResultBeta1[beta_idx];


    for (unsigned int stride = THREADS_PER_BLOCK / 2; stride > 1; stride >>= 1) {
        __syncthreads();
        if (t < stride) {
            partialSum[t] += partialSum[t+stride];
        }
    }

    __syncthreads();

    if (t == 0) {
        gpuResultBeta2[feature_beta_idx * num_features + blockIdx.x] = partialSum[0] + partialSum[1];
    }
}

//label and features are input variables, beta is the output variable
int train_gpu(int* labels, float* features, int num_features, double *beta)
{
    float* gpufeatures;
    int* gpulabels;
    double* gpuResultBeta1;
    double* gpuResultBeta2;
    double* gpubeta;
    int i, epochs;

    for(i = 0;i<9;i++)
    {
        //cout << features[i] << " ";
    }

    cudaMalloc((void**) &gpufeatures, num_features * FEAT_LEN * sizeof(float));
    cudaMemcpy(gpufeatures, features, num_features * FEAT_LEN * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void**) &gpulabels, num_features * sizeof(int));
    cudaMemcpy(gpulabels, labels,   num_features * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void**) &gpubeta, FEAT_LEN * sizeof(double));
    cudaMemcpy(gpubeta, beta,     FEAT_LEN * sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc((void**) &gpuResultBeta1, num_features * FEAT_LEN * sizeof(double));
    cudaMemset(gpuResultBeta1, 0,        num_features * FEAT_LEN * sizeof(double));

    cudaMalloc((void**) &gpuResultBeta2, num_features * FEAT_LEN * sizeof(double));
    cudaMemset(gpuResultBeta2, 0,        num_features * FEAT_LEN * sizeof(double));

    unsigned int numBlocks = num_features / THREADS_PER_BLOCK +1 ;

    for(epochs = 0; epochs < EPOCHS; epochs++)
    {
        // train one round to get all partial betas
        trainKernel<<< numBlocks, THREADS_PER_BLOCK >>>(gpulabels, gpufeatures, num_features, gpubeta, gpuResultBeta1);

        cudaThreadSynchronize();

        double* result = new double[num_features * FEAT_LEN * sizeof(double)];
        cudaMemcpy(result, gpuResultBeta1, num_features * FEAT_LEN * sizeof(double), cudaMemcpyDeviceToHost);
        cout << "gpu_reduce: ";
        double *sum = new double[FEAT_LEN];
        cout << "gputrain:";
        for (int j = 0; j < FEAT_LEN; j++)
        {
            for(i= 0; i < num_features; i++)
            {
                sum[j] += result[j*num_features + i];
            }
            cout << sum[j] <<" ";
        }
         cout << endl;


        // reduce all 9 betas for each feature
        for(i= 0; i < FEAT_LEN; i++)
        {
            reduceBetas<<< numBlocks, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(double) >>>(num_features , i, gpuResultBeta1, gpuResultBeta2);
        }
        cudaThreadSynchronize();
        for(i= 0; i < FEAT_LEN; i++)
        {
            reduceBetas<<< dim3(1), numBlocks, numBlocks * sizeof(double) >>>(num_features , i, gpuResultBeta2, gpuResultBeta1);
        }

        cudaThreadSynchronize();


        cudaMemcpy(result, gpuResultBeta1, num_features * FEAT_LEN * sizeof(double), cudaMemcpyDeviceToHost);
        cout << "gpu_reduce: ";
        for(i= 0; i < FEAT_LEN; i++)
        {
            cout << result[i*num_features] << " " ;
        }
        cout << "endl";

        cudaThreadSynchronize();
    }

    cudaMemcpy(beta, gpubeta, FEAT_LEN * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(gpufeatures);
    cudaFree(gpulabels);
    cudaFree(gpubeta);
    cudaFree(gpuResultBeta1);
    cudaFree(gpuResultBeta2);

    return 0;
}

__global__ void predictKernel(float* features, double* beta, int num_features, int* predictions)
{
    int x = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
    double z, proby;
    int pos;

    if (x  >= num_features) {
        return;
    }

    pos = x * FEAT_LEN ;

    for(int j = 0; j < FEAT_LEN; j++) {
            z += beta[j] * features[pos + j];
    }

    proby = (1.0 / (1.0 + exp(-z)));

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
int predict_gpu(float* features, double* beta, int num_features, int* predictions)
{
    float* gpufeatures;
    double* gpubeta;
    int* gpupredictions;

    std::cout << "starting gpu prediction" << endl;

    cudaMalloc((void**) &gpufeatures, num_features * FEAT_LEN * sizeof(float));
    cudaMemcpy(gpufeatures, features, num_features * FEAT_LEN * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void**) &gpubeta, FEAT_LEN * sizeof(double));
    cudaMemcpy(gpubeta, beta,     FEAT_LEN * sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc((void**) &gpupredictions, num_features * sizeof(int));
    cudaMemset(gpupredictions, 0,        num_features * sizeof(int));


    unsigned int numBlocks = num_features / THREADS_PER_BLOCK + 1 ;


    predictKernel<<< numBlocks, THREADS_PER_BLOCK >>>(gpufeatures, gpubeta, num_features, gpupredictions);

    cudaThreadSynchronize();

    cudaMemcpy(predictions, gpupredictions, num_features * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(gpufeatures);
    cudaFree(gpubeta);
    cudaFree(gpupredictions);

    return 0;

}
