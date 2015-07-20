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
#define THREADS_PER_BLOCK 512


#define MAX_THREADS 256

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)


__global__ void trainKernel(int* labels, float* features, int num_features, double *beta, double* gputmpbeta)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
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
__global__ void reduceBetas(int num_features, int numBlocks, int feature_beta_idx, double* gpuResultBeta1, double* gpuResultBeta2, double* gpubeta)
{
    extern __shared__ double partialSum[];

    int x = blockIdx.x * blockDim.x  + threadIdx.x;

    int beta_idx = feature_beta_idx * num_features + x;

    unsigned int t = threadIdx.x;

    if (x  >= num_features) {
        partialSum[t] = 0;
        return;
    }

    partialSum[t] = gpuResultBeta1[beta_idx];


    for (unsigned int stride = blockDim.x / 2 ; stride > 1; stride >>= 1) {
        __syncthreads();
        if (t < stride) {
            partialSum[t] += partialSum[t+stride];
        }
    }

    __syncthreads();

    //if numBlocks = 1 then save to gpubeta for next epoch
    if (t==0){
        if (numBlocks == 1)
        {
            gpubeta[feature_beta_idx * numBlocks + blockIdx.x] = partialSum[0] + partialSum[1];
        }
        else
        {
            gpuResultBeta2[feature_beta_idx * numBlocks + blockIdx.x] = partialSum[0] + partialSum[1];
        }
    }
}


//label and features are input variables, beta is the output variable
int train_gpu(int* labels, float* features, int num_features, double *beta)
{
    std::pair<float,float> p_correct, new_correct;
    int* cpulabels;
    double* beta_array;
    std::pair<float,float> * p_corr;
    float* gpufeatures;
    int* gpulabels;
    double* gpuResultBeta1;
    double* gpuResultBeta2;
    double* gpuResultBeta3;
    double* gpubeta;
    int i, epochs;

    unsigned int numBlocks = num_features / THREADS_PER_BLOCK +1;
    unsigned int numBlocks2 = numBlocks / THREADS_PER_BLOCK +1;

    std::cout << "numBlocks" << numBlocks;

    cudaMallocHost((void**) &cpulabels, num_features * sizeof(int));
    cudaMallocHost((void**) &beta_array, EPOCHS * FEAT_LEN * sizeof(double));
    cudaMallocHost((void**) &p_corr, EPOCHS * FEAT_LEN * sizeof(std::pair<float,float>));
    for(i = 0; i < num_features; i++)
    {
        cpulabels[i]=0;
    }

    cudaMalloc((void**) &gpufeatures, num_features * FEAT_LEN * sizeof(float));
    cudaMemcpy(gpufeatures, features, num_features * FEAT_LEN * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void**) &gpulabels, num_features * sizeof(int));
    cudaMemcpy(gpulabels, labels,   num_features * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void**) &gpubeta, FEAT_LEN * sizeof(double));
    cudaMemcpy(gpubeta, beta,     FEAT_LEN * sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc((void**) &gpuResultBeta1, num_features * FEAT_LEN * sizeof(double));
    cudaMemset(gpuResultBeta1, 0,        num_features * FEAT_LEN * sizeof(double));

    cudaMalloc((void**) &gpuResultBeta2, numBlocks * FEAT_LEN * sizeof(double));
    cudaMemset(gpuResultBeta2, 0,        numBlocks * FEAT_LEN * sizeof(double));

    cudaMalloc((void**) &gpuResultBeta3, numBlocks2 * FEAT_LEN * sizeof(double));
    cudaMemset(gpuResultBeta3, 0,        numBlocks2 * FEAT_LEN * sizeof(double));

    for(epochs = 0; epochs < EPOCHS; epochs++)
    {
        // train one round to get all partial betas
        trainKernel<<< numBlocks, THREADS_PER_BLOCK >>>(gpulabels, gpufeatures, num_features, gpubeta, gpuResultBeta1);


        // we need three reduce because otherwise there are too much threads per block because we have too much features
        //furthermore 3 tmp results on gpu are necessary because gpuResultBeta1 is needed in the next iteration and must not change
        // reduce all 9 betas for each feature
        for(i= 0; i < FEAT_LEN; i++)
        {
            reduceBetas<<< numBlocks, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(double) >>>(num_features, numBlocks, i, gpuResultBeta1, gpuResultBeta2, gpubeta);
        }

        // reduce all 9 betas for each feature
        for(i= 0; i < FEAT_LEN; i++)
        {
            reduceBetas<<< numBlocks2, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(double) >>>(numBlocks, numBlocks2, i, gpuResultBeta2, gpuResultBeta3, gpubeta);
        }

        for(i= 0; i < FEAT_LEN; i++)
        {
            reduceBetas<<< dim3(1), THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(double) >>>(numBlocks2 , 1, i, gpuResultBeta3, gpuResultBeta2, gpubeta);
        }


        cudaMemcpy(beta, gpubeta, FEAT_LEN * sizeof(double), cudaMemcpyDeviceToHost);
        predict_gpu(features,beta,num_features,cpulabels);
        p_corr[i] = calcPCorrect(labels, cpulabels, num_features);
        for(int j = 0; j < FEAT_LEN; j++) {

            beta_array[i*FEAT_LEN +j] = beta[j];
        }
    }

    //choose best beta
    bool break_outer =false;
    for(float f = 0.99 ; f > 0.5 ; f -=0.02)
    {
        for(int i = 0; i< EPOCHS; i++)
        {
            if(p_corr[i].first >= f && p_corr[i].second >= f)
            {
                for(int k = 0; k < FEAT_LEN; k++)
                {
                    beta[k] = beta_array[i*FEAT_LEN +k];
                }
                break_outer = true;
            }
        }
        if(break_outer)
        {
            break;
        }
    }

    cudaFree(gpufeatures);
    cudaFree(gpulabels);
    cudaFree(gpubeta);
    cudaFree(gpuResultBeta1);
    cudaFree(gpuResultBeta2);
    cudaFree(gpuResultBeta3);

    cudaFreeHost(cpulabels);

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


std::pair<float,float> calcPCorrect(int* labels, int* predicted, int num_features)
{
    float total_zeros = 0;
    float total_ones = 0;
    float correct_zeros = 0;
    float correct_ones = 0;

    for(int i = 0; i < num_features; i++) {
        if(labels[i] == predicted[i]) {
            if(labels[i] == 1)
            {
                correct_ones++;
            }
            else
            {
                correct_zeros++;
            }
        }
        if(labels[i] == 1)
        {
            total_ones++;
        }
        else
        {
            total_zeros++;
        }
    }

    return (std::make_pair((correct_ones/total_ones),(correct_zeros/ total_zeros)));
}


/*
int main(int argc, char *argv[])
{
    int num_features = 2;
    float* features = new float[18];
    features[0] = -0.215425;
    features[1]= -0.489831;
    features[2] = -0.733932;
    features[3] = -0.105368;
    features[4] = -0.197275;
    features[5] = -0.288388;
    features[6] = -0.370837;
    features[7] = -0.474699;
    features[8] = -0.538087;
    features[9] = -0.224642;
    features[10] = -0.49868;
    features[11] = -0.743322;
    features[12] = -0.106715;
    features[13] = -0.198606;
    features[14] = -0.29024;
    features[15] = -0.372176;
    features[16] = -0.476292;
    features[17] = -0.539666;
    int* labels = new int[2];
    labels[0] = 0;
    labels[1] = 0;


    double* beta = new double[9];
    train_gpu( labels, features, num_features, beta);
    //features =

}*/
