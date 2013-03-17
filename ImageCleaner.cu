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

__global__ void pre_compute(float *cos_term, float *sin_term)
{

//  int BlockIndex = blockIdx.x * blockDim.x;
//  int index = blockIdx.x * blockDim.x + threadIdx.x;

  float fft_real = 0;
  float fft_imag = 0;

  int tx =  threadIdx.x;

   __syncthreads();
      for(unsigned int n = 0; n < SIZEY; n++)
      {
        float term = -2 * PI * tx * n / SIZEY;
        fft_real = cos(term);
        fft_imag = sin(term);
  	cos_term[n*SIZEY + tx] = fft_real;
  	sin_term[n*SIZEY + tx] = fft_imag;
      }
   __syncthreads();
}

__global__ void pre_compute_i(float *cos_term, float *sin_term)
{

//  int BlockIndex = blockIdx.x * blockDim.x;
//  int index = blockIdx.x * blockDim.x + threadIdx.x;

  float fft_real = 0;
  float fft_imag = 0;

  int tx =  threadIdx.x;

   __syncthreads();
      for(unsigned int n = 0; n < SIZEY; n++)
      {
        float term = 2 * PI * tx * n / SIZEY;
        fft_real = cos(term);
        fft_imag = sin(term);
  	cos_term[n*SIZEY + tx] = fft_real;
  	sin_term[n*SIZEY + tx] = fft_imag;
      }
   __syncthreads();
}

__global__ void cpu_fftx_cuda_pre(float *real_image, float *imag_image, int size_x, int size_y, float *cos_term, float *sin_term)
{

//  int BlockIndex = blockIdx.x * blockDim.x;
//  int index = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float realInBuffer[SIZEY];
  __shared__ float imagInBuffer[SIZEY];

  __shared__ float fft_real_s[SIZEY];
  __shared__ float fft_imag_s[SIZEY];
//  __shared__ float realOutBuffer[SIZEY];
//  __shared__ float imagOutBuffer[SIZEY];

//  float fft_real = 0;
//  float fft_imag = 0;
  // Compute the value for this index
  float real_value = 0;
  float imag_value = 0;

//  float real_mul =0 ;
//  float imag_mul =0 ;

  int tx =  threadIdx.x;
  int bx =  blockIdx.x * SIZEY;
  int idx = bx + tx;


     realInBuffer[tx] = real_image[idx];
     imagInBuffer[tx] = imag_image[idx];

    	

      __syncthreads();


      for(unsigned int n = 0; n < SIZEY; n++)
      {
        fft_real_s[tx] = cos_term[n*SIZEY + tx];
        fft_imag_s[tx] = sin_term[n*SIZEY + tx];
//        __syncthreads();

        real_value += (realInBuffer[n] * fft_real_s[tx]) - (imagInBuffer[n] * fft_imag_s[tx]);
        imag_value += (imagInBuffer[n] * fft_real_s[tx]) + (realInBuffer[n] * fft_imag_s[tx]);
//        __syncthreads();
      }


      real_image[idx] = real_value;
      imag_image[idx] = imag_value;
//      __syncthreads();

}

__global__ void cpu_ifftx_cuda_pre(float *real_image, float *imag_image, int size_x, int size_y, float *cos_term, float *sin_term)
{

//  int BlockIndex = blockIdx.x * blockDim.x;
//  int index = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float realInBuffer[SIZEY];
  __shared__ float imagInBuffer[SIZEY];

  __shared__ float fft_real_s[SIZEY];
  __shared__ float fft_imag_s[SIZEY];
//  __shared__ float realOutBuffer[SIZEY];
//  __shared__ float imagOutBuffer[SIZEY];

//  float fft_real = 0;
//  float fft_imag = 0;
  // Compute the value for this index
  float real_value = 0;
  float imag_value = 0;

  int tx =  threadIdx.x;
  int bx =  blockIdx.x * SIZEY;
  int idx = bx + tx;


     realInBuffer[tx] = real_image[idx];
     imagInBuffer[tx] = imag_image[idx];

    	

      __syncthreads();


      for(unsigned int n = 0; n < SIZEY; n++)
      {
 //       float term = -2 * PI * threadIdx.x * n / SIZEY;
 //       fft_real = cos(term);
 //       fft_imag = sin(term);
        fft_real_s[tx] = cos_term[n*SIZEY + tx];
        fft_imag_s[tx] = sin_term[n*SIZEY + tx];
//        __syncthreads();

        real_value += (realInBuffer[n] * fft_real_s[tx]) - (imagInBuffer[n] * fft_imag_s[tx]);
        imag_value += (imagInBuffer[n] * fft_real_s[tx]) + (realInBuffer[n] * fft_imag_s[tx]);
//        __syncthreads();
      }

// Testing 
//	real_value = realInBuffer[tx] * 0.1;
//        imag_value = imagInBuffer[tx]* 5;

//      __syncthreads();

 //      realOutBuffer[threadIdx.x] = real_value;
 //      imagOutBuffer[threadIdx.x] = imag_value;

      real_image[idx] = real_value/SIZEY;
      imag_image[idx] = imag_value/SIZEY;
//      real_image[blockIdx.x*SIZEX + threadIdx.x] = realOutBuffer[threadIdx.x];
//      imag_image[blockIdx.x*SIZEX + threadIdx.x] = imagOutBuffer[threadIdx.x];
//      __syncthreads();

}

__global__ void cpu_ffty_cuda_pre(float *real_image, float *imag_image, int size_x, int size_y, float *cos_term, float *sin_term)
{

//  int BlockIndex = blockIdx.x * blockDim.x;
//  int index = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float realInBuffer[SIZEY];
  __shared__ float imagInBuffer[SIZEY];

  __shared__ float fft_real_s[SIZEY];
  __shared__ float fft_imag_s[SIZEY];
//  __shared__ float realOutBuffer[SIZEY];
//  __shared__ float imagOutBuffer[SIZEY];

//  float fft_real = 0;
//  float fft_imag = 0;
  // Compute the value for this index
  float real_value = 0;
  float imag_value = 0;

  int tx =  threadIdx.x;
  int bx =  blockIdx.x ;
  int idx = bx + tx*SIZEX;


     realInBuffer[tx] = real_image[idx];
     imagInBuffer[tx] = imag_image[idx];

    	

      __syncthreads();


      for(unsigned int n = 0; n < SIZEY; n++)
      {
        fft_real_s[tx] = cos_term[n*SIZEY + tx];
        fft_imag_s[tx] = sin_term[n*SIZEY + tx];

        real_value += (realInBuffer[n] * fft_real_s[tx]) - (imagInBuffer[n] * fft_imag_s[tx]);
        imag_value += (imagInBuffer[n] * fft_real_s[tx]) + (realInBuffer[n] * fft_imag_s[tx]);
//        __syncthreads();
      }


      real_image[idx] = real_value;
      imag_image[idx] = imag_value;
//      __syncthreads();

}

__global__ void cpu_iffty_cuda_pre(float *real_image, float *imag_image, int size_x, int size_y, float *cos_term, float *sin_term)
{

//  int BlockIndex = blockIdx.x * blockDim.x;
//  int index = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float realInBuffer[SIZEY];
  __shared__ float imagInBuffer[SIZEY];

  __shared__ float fft_real_s[SIZEY];
  __shared__ float fft_imag_s[SIZEY];
//  __shared__ float realOutBuffer[SIZEY];
//  __shared__ float imagOutBuffer[SIZEY];

//  float fft_real = 0;
//  float fft_imag = 0;
  // Compute the value for this index
  float real_value = 0;
  float imag_value = 0;

  int tx =  threadIdx.x;
  int bx =  blockIdx.x ;
  int idx = bx + tx*SIZEX;


     realInBuffer[tx] = real_image[idx];
     imagInBuffer[tx] = imag_image[idx];

    	

      __syncthreads();


      for(unsigned int n = 0; n < SIZEY; n++)
      {
        fft_real_s[tx] = cos_term[n*SIZEY + tx];
        fft_imag_s[tx] = sin_term[n*SIZEY + tx];

        real_value += (realInBuffer[n] * fft_real_s[tx]) - (imagInBuffer[n] * fft_imag_s[tx]);
        imag_value += (imagInBuffer[n] * fft_real_s[tx]) + (realInBuffer[n] * fft_imag_s[tx]);
//        __syncthreads();
      }


      real_image[idx] = real_value/SIZEX;
      imag_image[idx] = imag_value/SIZEX;
//      __syncthreads();

}
//__global__ void cpu_fftx_cuda_map(float *real_image, float *imag_image, int size_x, int size_y, float *real_map, float *imag_map)
//{
//  __shared__ float fft_real[SIZEX];
//  __shared__ float fft_imag[SIZEX];
//}

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

  if(threadIdx.x<SIZEY){
      realInBuffer[threadIdx.x] = real_image[blockIdx.x*SIZEX + threadIdx.x];
      imagInBuffer[threadIdx.x] = imag_image[blockIdx.x*SIZEX + threadIdx.x];
      __syncthreads();

      for(unsigned int n = 0; n < SIZEY; n++)
      {
	float term = 2 * PI * threadIdx.x * n / SIZEY;
	fft_real = cos(term);
	fft_imag = sin(term);

	real_value += (realInBuffer[n] * fft_real) - (imagInBuffer[n] * fft_imag);
	imag_value += (imagInBuffer[n] * fft_real) + (realInBuffer[n] * fft_imag);
      }

      real_image[blockIdx.x*SIZEX + threadIdx.x] = real_value/SIZEY;
      imag_image[blockIdx.x*SIZEX + threadIdx.x] = imag_value/SIZEY;
	
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

  if(threadIdx.x<SIZEX){
      realInBuffer[threadIdx.x] = real_image[threadIdx.x*SIZEX + blockIdx.x];
      imagInBuffer[threadIdx.x] = imag_image[threadIdx.x*SIZEX + blockIdx.x];
      __syncthreads();

      for(unsigned int n = 0; n < SIZEX; n++)
      {
        float term = -2 * PI * threadIdx.x * n / SIZEX;
        fft_real = cos(term);
        fft_imag = sin(term);

        real_value += (realInBuffer[n] * fft_real) - (imagInBuffer[n] * fft_imag);
        imag_value += (imagInBuffer[n] * fft_real) + (realInBuffer[n] * fft_imag);
      }

      real_image[threadIdx.x*SIZEX + blockIdx.x] = real_value;
      imag_image[threadIdx.x*SIZEX + blockIdx.x] = imag_value;
//      __syncthreads();
	
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


  if(threadIdx.x<SIZEX){
      realInBuffer[threadIdx.x] = real_image[threadIdx.x*SIZEX + blockIdx.x];
      imagInBuffer[threadIdx.x] = imag_image[threadIdx.x*SIZEX + blockIdx.x];
      __syncthreads();

      for(unsigned int n = 0; n < SIZEX; n++)
      {
        float term = 2 * PI * threadIdx.x * n / SIZEX;
        fft_real = cos(term);
        fft_imag = sin(term);

        real_value += (realInBuffer[n] * fft_real) - (imagInBuffer[n] * fft_imag);
        imag_value += (imagInBuffer[n] * fft_real) + (realInBuffer[n] * fft_imag);
      }

      real_image[threadIdx.x*SIZEY + blockIdx.x] = real_value/SIZEX;
      imag_image[threadIdx.x*SIZEY + blockIdx.x] = imag_value/SIZEX;
//      __syncthreads();
	
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

  // Custom measurement
//  cudaEvent_t start_me,stop_me;
//  float fftx = 0, ifftx = 0, filter = 0;

  CUDA_ERROR_CHECK(cudaEventCreate(&start));
  CUDA_ERROR_CHECK(cudaEventCreate(&stop));

  // Create a stream and initialize it
  cudaStream_t filterStream;
  CUDA_ERROR_CHECK(cudaStreamCreate(&filterStream));

  // Alloc space on the device
  float *device_real, *device_imag;
  CUDA_ERROR_CHECK(cudaMalloc((void**)&device_real, matSize));
  CUDA_ERROR_CHECK(cudaMalloc((void**)&device_imag, matSize));
  

  float *cos_t, *sin_t;
  CUDA_ERROR_CHECK(cudaMalloc((void**)&cos_t, matSize));
  CUDA_ERROR_CHECK(cudaMalloc((void**)&sin_t, matSize));
//  float *real_m, *imag_m;
//  CUDA_ERROR_CHECK(cudaMalloc((void**)&real_m, matSize));
//  CUDA_ERROR_CHECK(cudaMalloc((void**)&imag_m, matSize));

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
  //exampleKernel<<<1,128,0,filterStream>>>(device_real,device_imag,size_x,size_y);

//  CUDA_ERROR_CHECK(cudaEventCreate(&start_me));
//  CUDA_ERROR_CHECK(cudaEventCreate(&stop_me));
//

//  cpu_fftx_cuda_map<<<SIZEY,SIZEY,0,filterStream>>>(device_real,device_imag,size_x,size_y,real_m,imag_m);
//  cpu_fftx_cuda_reduce<<<SIZEY,SIZEX,0,filterStream>>>(device_real,device_imag,size_x,size_y,real_m,imag_m);
dim3 fft_dims;
fft_dims.x = SIZEY;
fft_dims.y = 1;

//  CUDA_ERROR_CHECK(cudaEventRecord(start_me,filterStream));
  pre_compute<<<1,SIZEY,0,filterStream>>>(cos_t, sin_t);
  cpu_fftx_cuda_pre<<<fft_dims,SIZEY,0,filterStream>>>(device_real,device_imag,size_x,size_y,cos_t,sin_t);
//  CUDA_ERROR_CHECK(cudaEventRecord(stop_me,filterStream));
//  CUDA_ERROR_CHECK(cudaEventSynchronize(stop_me));
//  CUDA_ERROR_CHECK(cudaEventElapsedTime(&fftx,start_me,stop_me));
//  printf(" Cuda FFTx execution time: %f ms\n", fftx);


  cpu_ffty_cuda_pre<<<SIZEX,SIZEY,0,filterStream>>>(device_real,device_imag,size_x,size_y,cos_t,sin_t);
  //cpu_ffty_cuda<<<SIZEY,SIZEX,0,filterStream>>>(device_real,device_imag,size_x,size_y);
  cpu_filter_cuda<<<SIZEX,SIZEY,0,filterStream>>>(device_real,device_imag,size_x,size_y);
  pre_compute_i<<<1,SIZEY,0,filterStream>>>(cos_t, sin_t);
  cpu_ifftx_cuda_pre<<<SIZEX,SIZEY,0,filterStream>>>(device_real,device_imag,size_x,size_y,cos_t,sin_t);
  cpu_iffty_cuda_pre<<<SIZEX,SIZEY,0,filterStream>>>(device_real,device_imag,size_x,size_y,cos_t,sin_t);


  //cpu_ifftx_cuda<<<SIZEX,SIZEY,0,filterStream>>>(device_real,device_imag,size_x,size_y);
  //cpu_iffty_cuda<<<SIZEY,SIZEX,0,filterStream>>>(device_real,device_imag,size_x,size_y);

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
  CUDA_ERROR_CHECK(cudaFree(cos_t));
  CUDA_ERROR_CHECK(cudaFree(sin_t));

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

