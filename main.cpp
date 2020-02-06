/////////////////////////////////////////////////////////////////////////////
// Name:         main.cpp
// Purpose:      Simulating the discrete element model on a GPU
// Author:       Mae Woods UCL
// Modified by:
// Created:      Jan/2014
// Copyright:    Open source
// Licence:      None
// Dependencies: CUDA see http://www.nvidia.co.uk/object/cuda-parallel-computing-uk.html
/////////////////////////////////////////////////////////////////////////////

// ==========================================================================
// headers, declarations, constants
// ==========================================================================

#include <cstdlib>
#include <cstdio>
#include <sdkHelper.h>
#include <unistd.h>
#include <fcntl.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <iomanip>
#include <math.h>
#include <cuda.h>
#include "device_functions.h"
#include <curand_kernel.h>
#include <cutil_inline.h>
#include <iostream>
#include <fstream>
#include "main.h"
#include <iomanip>
#include "HostMemoryAlloc.h"
#include <cutil_inline.h>
#include <cstdlib>
#include <cstdio>
#include <string.h>

				//Compute capability 2.x architectures
#define CUPRINTF(fmt, ...) printf(fmt,	__VA_ARGS__)


using namespace std;
extern "C" void cudaInit(int argc, char **argv);

HostMemoryAlloc *hmalloc;

unsigned int timer;

#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)

inline void __checkCudaErrors(cudaError err, const char *file, const int line )
{
    if(cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",file, line, (int)err, cudaGetErrorString( err ) );
        exit(-1);        
    }
}

// This will output the proper error string when calling cudaGetLastError
#define getLastCudaError(msg)      __getLastCudaError (msg, __FILE__, __LINE__)

inline void __getLastCudaError(const char *errorMessage, const char *file, const int line )
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
        file, line, errorMessage, (int)err, cudaGetErrorString( err ) );
        exit(-1);
    }
}

// General GPU Device CUDA Initialization
int gpuDeviceInit(int devID)
{
    int deviceCount;
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0)
    {
        fprintf(stderr, "gpuDeviceInit() CUDA error: no devices supporting CUDA.\n");
        exit(-1);
    }

    if (devID < 0)
       devID = 0;
        
    if (devID > deviceCount-1)
    {
        fprintf(stderr, "\n");
        fprintf(stderr, ">> %d CUDA capable GPU device(s) detected. <<\n", deviceCount);
        fprintf(stderr, ">> gpuDeviceInit (-device=%d) is not a valid GPU device. <<\n", devID);
        fprintf(stderr, "\n");
        return -devID;
    }

    cudaDeviceProp deviceProp;
    checkCudaErrors( cudaGetDeviceProperties(&deviceProp, devID) );

    if (deviceProp.major < 1)
    {
        fprintf(stderr, "gpuDeviceInit(): GPU device does not support CUDA.\n");
        exit(-1);                                                  
    }
    
    checkCudaErrors( cudaSetDevice(devID) );
    //printf("gpuDeviceInit() CUDA Device [%d]: \"%s\n", devID, deviceProp.name);

    return devID;
}

#define MAX(a,b) ((a > b) ? a : b)

// This function returns the best GPU (with maximum GFLOPS)
int gpuGetMaxGflopsDeviceId()
{
    int current_device     = 0, sm_per_multiproc  = 0;
    int max_compute_perf   = 0, max_perf_device   = 0;
    int device_count       = 0, best_SM_arch      = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceCount( &device_count );
    
    // Find the best major SM Architecture GPU device
    while (current_device < device_count)
    {
        cudaGetDeviceProperties( &deviceProp, current_device );
        if (deviceProp.major > 0 && deviceProp.major < 9999)
        {
            best_SM_arch = MAX(best_SM_arch, deviceProp.major);
        }
        current_device++;
    }

    // Find the best CUDA capable GPU device
    current_device = 0;
    while( current_device < device_count )
    {
        cudaGetDeviceProperties( &deviceProp, current_device );
        if (deviceProp.major == 9999 && deviceProp.minor == 9999)
        {
            sm_per_multiproc = 1;
        }
        else
        {
            sm_per_multiproc = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
        }
        
        int compute_perf  = deviceProp.multiProcessorCount * sm_per_multiproc * deviceProp.clockRate;
        
    if( compute_perf  > max_compute_perf )
    {
            // If we find GPU with SM major > 2, search only these
            if ( best_SM_arch > 2 )
            {
                // If our device==dest_SM_arch, choose this, or else pass
                if (deviceProp.major == best_SM_arch)
                {
                    max_compute_perf  = compute_perf;
                    max_perf_device   = current_device;
                 }
            }
            else
            {
                max_compute_perf  = compute_perf;
                max_perf_device   = current_device;
             }
        }
        ++current_device;
    }
    return max_perf_device;
}

int getRunNo(int argc, const char **argv)
{

int Rint = 0;

if (checkCmdLineFlag(argc, argv, "run"))
    {
        Rint = getCmdLineArgumentInt(argc, argv, "run=");

}


return Rint;
}

int findCudaDevice(int argc, const char **argv)
{
    cudaDeviceProp deviceProp;
    int devID = 0;
    // If the command-line has a device number specified, use it
    if (checkCmdLineFlag(argc, argv, "device"))
    {
        devID = getCmdLineArgumentInt(argc, argv, "device=");
        if (devID < 0)
        {
            exit(-1);
        }
        else
        {
            devID = gpuDeviceInit(devID);
            if (devID < 0)
            {
                exit(-1);
            }
        }
    }
    else
    {
        // Otherwise pick the device with highest Gflops/s
        devID = gpuGetMaxGflopsDeviceId();
        checkCudaErrors( cudaSetDevice( devID ) );
        checkCudaErrors( cudaGetDeviceProperties(&deviceProp, devID) );
    }
    return devID;
}

int main(int argc, char** argv) 
{

    
    int devID;
	cudaDeviceProp props;
	CUdevice cudaDevice;

	//Initialize cuda (pass in device number on cmd line to select GPU, e.g. ./particles -device=4
	//check for a command line argument that specifies a GPU (this is optional)
	devID = findCudaDevice((const int)argc, (const char **)argv);
    
    checkCudaErrors(cudaGetDevice(&devID));
	checkCudaErrors(cudaGetDeviceProperties(&props, devID));

     // Set the number of cells here
    int numThreads=50;
    // Set the number of independent simulations
    int numBlocks=10;

	int runNum = 0;

	//Check if this is the first simulation, or a repeat using input data.
	runNum = getRunNo((const int)argc, (const char **)argv);

    //initialize class and set memory allocation on device
    hmalloc = new HostMemoryAlloc(numBlocks, numThreads); 
	
    hmalloc->Initialise(runNum);

    hmalloc->randkernel(numThreads, numBlocks);

	//If you want to print kernel data to file use cuPrintf
	cudaPrintfInit();

	hmalloc->XData(numThreads, numBlocks, runNum);

	cudaPrintfDisplay(stdout, 2000000000);

        cudaPrintfEnd();

	hmalloc->copyAfromD(runNum);


cudaThreadExit();
cudaDeviceReset();

    return 0;
}

