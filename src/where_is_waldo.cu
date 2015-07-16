/*
 ============================================================================
 Name        : where_is_waldo.cu
 Author      : volk, hettich
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA compute reciprocals
 ============================================================================
 */
#include "where_is_waldo.h"
#include "PPM.hh"

#define MAX_THREADS 512

//static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
//#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

# define M_PI           3.14159265358979323846  /* pi */

/**
 * CUDA kernel that computes reciprocal values for a given vector
 */
/*__global__ void reciprocalKernel(float *data, unsigned vectorSize) {
	unsigned idx = blockIdx.x*blockDim.x+threadIdx.x;
	if (idx < vectorSize)
		data[idx] = 1.0/data[idx];
}*/

__global__ void gaussKernel(const float *_src, float *_dst, float *matrix, int width, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y;
    //int pos = y * _w + x;

    if (x >= w || y >= h) {
        return;
    }

    int half = width / 2;

    float blurRed = 0.f;
	float blurGreen = 0.f;
	float blurBlue = 0.f;

	for (int _y = -half; _y <= half; _y++) {
		for (int _x = -half; _x <= half; _x++) {
			int tmpX = x+_x;
			int tmpY = y+_y;

			if (tmpX >= w || tmpX < 0) {
				tmpX = x;
			}

			if (tmpY >= h || tmpY < 0) {
				tmpY = y;
			}

			int pos2 = tmpX * 3 + w * 3 * tmpY;

			float pixelRed = _src[pos2];

			float pixelGreen = _src[pos2+1];

			float pixelBlue = _src[pos2+2];

			int idx = (_y + half) * width + (_x + half);
			float weight = matrix[idx];

			blurRed += pixelRed * weight;
			blurGreen += pixelGreen * weight;
			blurBlue += pixelBlue * weight;
		}
	}

	int pos = x * 3 + w * 3 * y;
	_dst[pos] = blurRed;
	_dst[pos+1] = blurGreen;
	_dst[pos+2] = blurBlue;
}


/*float *gpuReciprocal(float *data, unsigned size)
{
	float *rc = new float[size];
	float *gpuData;

	CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuData, sizeof(float)*size));
	CUDA_CHECK_RETURN(cudaMemcpy(gpuData, data, sizeof(float)*size, cudaMemcpyHostToDevice));
	
	static const int BLOCK_SIZE = 256;
	const int blockCount = (size+BLOCK_SIZE-1)/BLOCK_SIZE;
	reciprocalKernel<<<blockCount, BLOCK_SIZE>>> (gpuData, size);

	CUDA_CHECK_RETURN(cudaMemcpy(rc, gpuData, sizeof(float)*size, cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaFree(gpuData));
	return rc;
}*/

int doGauss(const char* imagePath, const char* outputPath)
{
    /** BEGIN init kernel. */
    float* img;
    float* img2;

    int w, h;
    // read imagePath (input image).
    ppm::readPPM(imagePath, w, h, &img);
    ppm::readPPM(imagePath, w, h, &img2);

    int nPix = w * h;
    float* gpuImg;
    float* gpuResult;
    float* gpuMatrix;

    const int width = 9;

    cudaMalloc((void**) &gpuImg, nPix * 3 * sizeof(float));
    cudaMalloc((void**) &gpuResult, nPix * 3 * sizeof(float));
    cudaMalloc((void**) &gpuMatrix, width * width * sizeof(float));

    cudaMemcpy(gpuImg, img, nPix * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(gpuResult, 0, nPix * 3 * sizeof(float));
    /** END init kernel. */

    /** BEGIN create gaussian */
    const float sigma = 10.f;

    const int half = width/2;
    float sum = 0.f;

    float* matrix = (float *) malloc(81 * sizeof(float));


    for (int y = -half; y <= half; y++) {
        for (int x = -half; x <= half; x++) {
            float weight = std::exp(-(static_cast<float>(x * x + y * y)/(2.f * sigma * sigma)));
        	int idx = (y + half) * width + x + half;

            matrix[idx] = weight;
            sum += weight;
        }
    }

    float normal = 1.f/sum;

    int idx;

    for (int y = -half; y <= half; y++) {
        for (int x = -half; x <= half; x++) {
            idx = (y + half) * width + x + half;

            matrix[idx] = matrix[idx] * normal;
            //std::cout << matrix[idx] << "  ";
        }
        //std::cout << std::endl;
    }

    cudaMemcpy(gpuMatrix, matrix, width * width * sizeof(float), cudaMemcpyHostToDevice);
    /** END create gaussian */

    /** BEGIN run kernel. */
    dim3 threadBlock(MAX_THREADS);
    dim3 blockGrid(w / MAX_THREADS + 1, h, 1);

    // run gaussian blur kernel.
    std::cout<<"would execute kernerl. But skipping an using cpu part at the moment."<<std::endl;
    gaussKernel<<< blockGrid, threadBlock >>>(gpuImg, gpuResult, gpuMatrix, width, w, h);

    /*for (int x = 0; x < w; x++) {
    	for (int y = 0; y < h; y++) {
    		float blurRed = 0.f;
			float blurGreen = 0.f;
			float blurBlue = 0.f;

			for (int _y = -half; _y <= half; _y++) {
				for (int _x = -half; _x <= half; _x++) {
					int tmpX = x+_x;
					int tmpY = y+_y;

					if (tmpX >= w || tmpX < 0) {
						tmpX = x;
					}

					if (tmpY >= h || tmpY < 0) {
						tmpY = y;
					}

					int pos2 = tmpX * 3 + w * 3 * tmpY;

					float pixelRed = img[pos2];

					float pixelGreen = img[pos2+1];

					float pixelBlue = img[pos2+2];

					int idx = (_y + half) * width + (_x + half);
					float weight = matrix[idx];

					blurRed += pixelRed * weight;
					blurGreen += pixelGreen * weight;
					blurBlue += pixelBlue * weight;
				}
			}

			int pos = x * 3 + w * 3 * y;
			img2[pos] = blurRed;
			img2[pos+1] = blurGreen;
			img2[pos+2] = blurBlue;
    	}
    }*/
    /** END run kernel. */
    //std::cout << "finisehd cpu kernel" << std::endl;
    /** BEGIN save to disk */
    cudaMemcpy(img2, gpuResult, nPix * 3 * sizeof(float), cudaMemcpyDeviceToHost);

    ppm::writePPM(outputPath, w, h, (float*) img2);

    std::cout << "finisehd writing image." << std::endl;

    cudaFree(gpuResult);
    cudaFree(gpuImg);
    delete[] img;
    /** BEGIN save to disk */

    return 0;
}

/*int main(int argc, char *argv[])
{
    const char* a = "build-where_is_waldo-Desktop-Debug/training_areas.ppm";
    const char* b = "build-where_is_waldo-Desktop-Debug/training_areas_output.ppm";
    std::cout << "Start Gauss" << std::endl;
    doGauss(a, b);
    std::cout << "End Gauss" << std::endl;
}*/

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
/*static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (1);
}*/
