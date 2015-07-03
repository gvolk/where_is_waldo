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

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

# define M_PI           3.14159265358979323846  /* pi */

/**
 * CUDA kernel that computes reciprocal values for a given vector
 */
/*__global__ void reciprocalKernel(float *data, unsigned vectorSize) {
	unsigned idx = blockIdx.x*blockDim.x+threadIdx.x;
	if (idx < vectorSize)
		data[idx] = 1.0/data[idx];
}*/

__global__ void gaussKernel(float *_src, float *_dst, float* _weight, int _width, int _w, int _h) {
    int _x = blockIdx.x * blockDim.x + threadIdx.x;
    int _y = blockIdx.y * blockDim.y + threadIdx.y;
    //int pos = y * _w + x;

    if (_x >= _w || _y >= _h) {
        return;
    }

    int half = _width/2;
    float blur = 0.f;

    for (int y = -half; y <= half; y++) {
        for (int x = -half; x <= half; x++) {
            int w = min(max(_x+x, 0), _w);
            int h = min(max(_y+y, 0), _h);

            int idx = w + _w*h;

            float pixel = _src[idx];

            idx = (y+half)*_width + x+half;

            float weight = _weight[idx];

            blur += pixel*weight;
        }
    }

    _dst[_x + _y*_w] = blur;
}

/**
 * Host function that copies the data and launches the work on GPU
 */
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
}

float *cpuReciprocal(float *data, unsigned size)
{
	float *rc = new float[size];
	for (unsigned cnt = 0; cnt < size; ++cnt) rc[cnt] = 1.0/data[cnt];
	return rc;
}


void initialize(float *data, unsigned size)
{
	for (unsigned i = 0; i < size; ++i)
		data[i] = .5*(i+1);
}*/

int run(char* imagePath, char* outputPath)
{
    /** BEGIN init kernel. */
    float* img;

    int w, h;
    // read imagePath (input image).
    ppm::readPPM(imagePath, w, h, &img);

    int nPix = w*h;
    float* gpuImg;
    float* gpuResult;

    cudaMalloc((void**) &gpuImg, nPix*3*sizeof(float));
    cudaMalloc((void**) &gpuResult, nPix*3*sizeof(float));

    cudaMemcpy(gpuImg, img, nPix*3*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(gpuResult, 0, nPix*3*sizeof(float));
    /** END init kernel. */

    /** BEGIN create gaussian */
    const int width = 9;
    const float sigma = 2.f;

    const int half = width/2;
    float sum = 0.f;

    /*vector<float> matrix;
    matrix.resize(width*width);*/
    float* matrix;

    for (int y = -half; y <= half; y++) {
        for (int x = -half; x <= half; x++) {
            float weight = std::exp(-static_cast<float>(x*x+y*y)/(2.f*sigma*sigma));
            int idx = (y+half)*width + x+half;

            matrix[idx] = weight;
            sum += weight;
        }
    }

    float normal = 1.f/sum;

    for (int y = -half; y <= half; y++) {
        for (int x = -half; x <= half; x++) {
            int idx = (y+half)*width + x+half;

            matrix[idx] *= normal;
        }
    }
    /** END create gaussian */

    /** BEGIN run kernel. */
    static const int BLOCK_WIDTH = 32;
    int x = w/BLOCK_WIDTH;
    int y = h/BLOCK_WIDTH;

    const dim3 grid (x, y, 1);
    const dim3 block (BLOCK_WIDTH, BLOCK_WIDTH, 1);

    // run gaussian blur kernel.
    std::cout<<"would execute kernerl."<<std::endl;
    gaussKernel<<< grid, block >>>(gpuImg, gpuResult, matrix, width, w, h);
    /** END run kernel. */

    /** BEGIN save to disk */
    cudaMemcpy(img, gpuResult, nPix*3*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(gpuResult);
    cudaFree(gpuImg);

    ppm::writePPM(outputPath, w, h, (float*) img);

    delete[] img;
    /** BEGIN save to disk */

    std::cout<<"done"<<std::endl;

    return 0;

    /*static const int WORK_SIZE = 65530;
	float *data = new float[WORK_SIZE];

	initialize (data, WORK_SIZE);

	float *recCpu = cpuReciprocal(data, WORK_SIZE);
	float *recGpu = gpuReciprocal(data, WORK_SIZE);
	float cpuSum = std::accumulate (recCpu, recCpu+WORK_SIZE, 0.0);
	float gpuSum = std::accumulate (recGpu, recGpu+WORK_SIZE, 0.0);

    // Verify the results
	std::cout<<"gpuSum2 = "<<gpuSum<< " cpuSum = " <<cpuSum<<std::endl;

    // Free memory
	delete[] data;
	delete[] recCpu;
	delete[] recGpu;

    return 0;*/
}

/*
int main(int argc, char *argv[])
{
	string imagePath;
	string outputPath;
	/*
	// load image to perform gaussian blur.
		if (argc > 1) {
			imagePath = string(argv[1]);
			if (argc > 2) {
				outputPath = string(argv[2]);
			} else {
				outputPath = "output.jpg";
			}
		} else {
			cerr << "Please provide an input (and output) image path as arguments to this application." << endl;
			exit(1);
		}
	return run(imagePath, outputPath);
	run(imagePath,outputPath);
}
*/


/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (1);
}

