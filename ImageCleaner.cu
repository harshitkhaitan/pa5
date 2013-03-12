#include "ImageCleaner.h"

#ifndef SIZEX
#error Please define SIZEX.
#endif
#ifndef SIZEY
#error Please define SIZEY.
#endif

#define PI      3.14159256

//----------------------------------------------------------------
// TODO:  CREATE NEW KERNELS HERE.  YOU CAN PLACE YOUR CALLS TO
//        THEM IN THE INDICATED SECTION INSIDE THE 'filterImage'
//        FUNCTION.
//
// BEGIN ADD KERNEL DEFINITIONS
//----------------------------------------------------------------


__global__ void exampleKernel(float *real_image, float *imag_image, int size_x, int size_y)
{
  // Currently does nothing
}

__global__ void cpu_fftx_cuda(float *real_image, float *imag_image, int size_x, int size_y)
{

//  int BlockIndex = blockIdx.x * blockDim.x;
//  int index = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float realInBuffer[SIZEY];
  __shared__ float imagInBuffer[SIZEY];
  float fft_real;
  float fft_imag;
  // Compute the value for this index
  float real_value = 0;
  float imag_value = 0;


  if(threadIdx.x<size_y){
      realInBuffer[threadIdx.x] = real_image[blockIdx.x*size_x + threadIdx.x];
      imagInBuffer[threadIdx.x] = imag_image[blockIdx.x*size_x + threadIdx.x];
      __syncthreads();
      for(unsigned int n = 0; n < size_y; n++)
      {
        float term = -2 * PI * threadIdx.x * n / size_y;
        fft_real = cos(term);
        fft_imag = sin(term);

        real_value += (realInBuffer[n] * fft_real) - (imagInBuffer[n] * fft_imag);
        imag_value += (imagInBuffer[n] * fft_real) + (realInBuffer[n] * fft_imag);
      }

      real_image[blockIdx.x*size_x + threadIdx.x] = real_value;
      imag_image[blockIdx.x*size_x + threadIdx.x] = imag_value;
      __syncthreads();

//     printf("Block Idx %d \n", blockIdx.x);
//      printf("Block DIM %d \n", blockDim.x);
//      printf("Thread ID %d \n", threadIdx.x);
	
  }	

}

__global__ void cpu_ifftx_cuda(float *real_image, float *imag_image, int size_x, int size_y)
{

//  int BlockIndex = blockIdx.x * blockDim.x;
//  int index = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float realInBuffer[SIZEY];
  __shared__ float imagInBuffer[SIZEY];
 // Compute the value for this index
 float real_value = 0;
 float imag_value = 0;
      float fft_real;
      float fft_imag;

  if(threadIdx.x<size_y){
      realInBuffer[threadIdx.x] = real_image[blockIdx.x*size_x + threadIdx.x];
      imagInBuffer[threadIdx.x] = imag_image[blockIdx.x*size_x + threadIdx.x];
      __syncthreads();

      for(unsigned int n = 0; n < size_y; n++)
      {
	float term = -2 * PI * threadIdx.x * n / size_y;
	fft_real = cos(term);
	fft_imag = sin(term);

	real_value += (realInBuffer[n] * fft_real) - (imagInBuffer[n] * fft_imag);
	imag_value += (imagInBuffer[n] * fft_real) + (realInBuffer[n] * fft_imag);
      }

      real_image[blockIdx.x*size_x + threadIdx.x] = real_value/size_y;
      imag_image[blockIdx.x*size_x + threadIdx.x] = imag_value/size_y;
      __syncthreads();

//     printf("Block Idx %d \n", blockIdx.x);
//      printf("Block DIM %d \n", blockDim.x);
//      printf("Thread ID %d \n", threadIdx.x);
	
  }	

}

__global__ void cpu_ffty_cuda(float *real_image, float *imag_image, int size_x, int size_y)
{

//  int BlockIndex = blockIdx.x * blockDim.x;
//  int index = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float realInBuffer[SIZEY];
  __shared__ float imagInBuffer[SIZEY];
      float fft_real;
      float fft_imag;
      // Compute the value for this index
      float real_value = 0;
      float imag_value = 0;

  if(threadIdx.x<size_x){
      realInBuffer[threadIdx.x] = real_image[threadIdx.x*size_x + blockIdx.x];
      imagInBuffer[threadIdx.x] = imag_image[threadIdx.x*size_x + blockIdx.x];
      __syncthreads();

      for(unsigned int n = 0; n < size_x; n++)
      {
        float term = -2 * PI * threadIdx.x * n / size_x;
        fft_real = cos(term);
        fft_imag = sin(term);

        real_value += (realInBuffer[n] * fft_real) - (imagInBuffer[n] * fft_imag);
        imag_value += (imagInBuffer[n] * fft_real) + (realInBuffer[n] * fft_imag);
      }

      real_image[threadIdx.x*size_x + blockIdx.x] = real_value;
      imag_image[threadIdx.x*size_x + blockIdx.x] = imag_value;
      __syncthreads();
	
  }	

}

__global__ void cpu_iffty_cuda(float *real_image, float *imag_image, int size_x, int size_y)
{

//  int BlockIndex = blockIdx.x * blockDim.x;
//  int index = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float realInBuffer[SIZEY];
  __shared__ float imagInBuffer[SIZEY];
      float fft_real;
      float fft_imag;
      // Compute the value for this index
      float real_value = 0;
      float imag_value = 0;


  if(threadIdx.x<size_x){
      realInBuffer[threadIdx.x] = real_image[threadIdx.x*size_x + blockIdx.x];
      imagInBuffer[threadIdx.x] = imag_image[threadIdx.x*size_x + blockIdx.x];
      __syncthreads();

      for(unsigned int n = 0; n < size_x; n++)
      {
        float term = -2 * PI * threadIdx.x * n / size_x;
        fft_real = cos(term);
        fft_imag = sin(term);

        real_value += (realInBuffer[n] * fft_real) - (imagInBuffer[n] * fft_imag);
        imag_value += (imagInBuffer[n] * fft_real) + (realInBuffer[n] * fft_imag);
      }

      real_image[threadIdx.x*size_y + blockIdx.x] = real_value/size_x;
      imag_image[threadIdx.x*size_y + blockIdx.x] = imag_value/size_x;
      __syncthreads();
	
  }	

}
__global__ void cpu_filter_cuda(float *real_image, float *imag_image, int size_x, int size_y)
{
  int eightX = size_x/8;
  int eight7X = size_x - eightX;
  int eightY = size_y/8;
  int eight7Y = size_y - eightY;

  __syncthreads();
  if(!(blockIdx.x < eightX && threadIdx.x < eightY) &&
         !(blockIdx.x < eightX && threadIdx.x >= eight7Y) &&
         !(blockIdx.x >= eight7X && threadIdx.x < eightY) &&
         !(blockIdx.x >= eight7X && threadIdx.x >= eight7Y))
      {
        // Zero out these values
        real_image[threadIdx.x*size_x + blockIdx.x] = 0;
        imag_image[threadIdx.x*size_x + blockIdx.x] = 0;
      }
   __syncthreads();

}

//----------------------------------------------------------------
// END ADD KERNEL DEFINTIONS
//----------------------------------------------------------------

__host__ float filterImage(float *real_image, float *imag_image, int size_x, int size_y)
{
  // check that the sizes match up
  assert(size_x == SIZEX);
  assert(size_y == SIZEY);

  int matSize = size_x * size_y * sizeof(float);

  // These variables are for timing purposes
  float transferDown = 0, transferUp = 0, execution = 0;
  cudaEvent_t start,stop;

  CUDA_ERROR_CHECK(cudaEventCreate(&start));
  CUDA_ERROR_CHECK(cudaEventCreate(&stop));

  // Create a stream and initialize it
  cudaStream_t filterStream;
  CUDA_ERROR_CHECK(cudaStreamCreate(&filterStream));

  // Alloc space on the device
  float *device_real, *device_imag;
  CUDA_ERROR_CHECK(cudaMalloc((void**)&device_real, matSize));
  CUDA_ERROR_CHECK(cudaMalloc((void**)&device_imag, matSize));

  // Start timing for transfer down
  CUDA_ERROR_CHECK(cudaEventRecord(start,filterStream));
  
  // Here is where we copy matrices down to the device 
  CUDA_ERROR_CHECK(cudaMemcpy(device_real,real_image,matSize,cudaMemcpyHostToDevice));
  CUDA_ERROR_CHECK(cudaMemcpy(device_imag,imag_image,matSize,cudaMemcpyHostToDevice));
  
  // Stop timing for transfer down
  CUDA_ERROR_CHECK(cudaEventRecord(stop,filterStream));
  CUDA_ERROR_CHECK(cudaEventSynchronize(stop));
  CUDA_ERROR_CHECK(cudaEventElapsedTime(&transferDown,start,stop));

  // Start timing for the execution
  CUDA_ERROR_CHECK(cudaEventRecord(start,filterStream));

  //----------------------------------------------------------------
  // TODO: YOU SHOULD PLACE ALL YOUR KERNEL EXECUTIONS
  //        HERE BETWEEN THE CALLS FOR STARTING AND
  //        FINISHING TIMING FOR THE EXECUTION PHASE
  // BEGIN ADD KERNEL CALLS
  //----------------------------------------------------------------

  // This is an example kernel call, you should feel free to create
  // as many kernel calls as you feel are needed for your program
  // Each of the parameters are as follows:
  //    1. Number of thread blocks, can be either int or dim3 (see CUDA manual)
  //    2. Number of threads per thread block, can be either int or dim3 (see CUDA manual)
  //    3. Always should be '0' unless you read the CUDA manual and learn about dynamically allocating shared memory
  //    4. Stream to execute kernel on, should always be 'filterStream'
  //
  // Also note that you pass the pointers to the device memory to the kernel call
  exampleKernel<<<1,128,0,filterStream>>>(device_real,device_imag,size_x,size_y);
  cpu_fftx_cuda<<<SIZEX,SIZEY,0,filterStream>>>(device_real,device_imag,size_x,size_y);
  cpu_ffty_cuda<<<SIZEY,SIZEX,0,filterStream>>>(device_real,device_imag,size_x,size_y);
  cpu_filter_cuda<<<SIZEX,SIZEY,0,filterStream>>>(device_real,device_imag,size_x,size_y);
  cpu_ifftx_cuda<<<SIZEX,SIZEY,0,filterStream>>>(device_real,device_imag,size_x,size_y);
  cpu_iffty_cuda<<<SIZEY,SIZEX,0,filterStream>>>(device_real,device_imag,size_x,size_y);

  //---------------------------------------------------------------- 
  // END ADD KERNEL CALLS
  //----------------------------------------------------------------

  // Finish timimg for the execution 
  CUDA_ERROR_CHECK(cudaEventRecord(stop,filterStream));
  CUDA_ERROR_CHECK(cudaEventSynchronize(stop));
  CUDA_ERROR_CHECK(cudaEventElapsedTime(&execution,start,stop));

  // Start timing for the transfer up
  CUDA_ERROR_CHECK(cudaEventRecord(start,filterStream));

  // Here is where we copy matrices back from the device 
  CUDA_ERROR_CHECK(cudaMemcpy(real_image,device_real,matSize,cudaMemcpyDeviceToHost));
  CUDA_ERROR_CHECK(cudaMemcpy(imag_image,device_imag,matSize,cudaMemcpyDeviceToHost));

  // Finish timing for transfer up
  CUDA_ERROR_CHECK(cudaEventRecord(stop,filterStream));
  CUDA_ERROR_CHECK(cudaEventSynchronize(stop));
  CUDA_ERROR_CHECK(cudaEventElapsedTime(&transferUp,start,stop));

  // Synchronize the stream
  CUDA_ERROR_CHECK(cudaStreamSynchronize(filterStream));
  // Destroy the stream
  CUDA_ERROR_CHECK(cudaStreamDestroy(filterStream));
  // Destroy the events
  CUDA_ERROR_CHECK(cudaEventDestroy(start));
  CUDA_ERROR_CHECK(cudaEventDestroy(stop));

  // Free the memory
  CUDA_ERROR_CHECK(cudaFree(device_real));
  CUDA_ERROR_CHECK(cudaFree(device_imag));

  // Dump some usage statistics
  printf("CUDA IMPLEMENTATION STATISTICS:\n");
  printf("  Host to Device Transfer Time: %f ms\n", transferDown);
  printf("  Kernel(s) Execution Time: %f ms\n", execution);
  printf("  Device to Host Transfer Time: %f ms\n", transferUp);
  float totalTime = transferDown + execution + transferUp;
  printf("  Total CUDA Execution Time: %f ms\n\n", totalTime);
  // Return the total time to transfer and execute
  return totalTime;
}

