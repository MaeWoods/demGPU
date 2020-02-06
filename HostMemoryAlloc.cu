/*
 *  HostMemoryAlloc.cu
 *  
 *  Created by Mae Woods on 19/01/2012.
 *  Licensed under the GNU GPL. This file is free to
 *  use or modify.
 *  Created as part of a PhD project in computational
 *  biology in the department of CoMPLEX UCL
 *
 */

#include <cutil_inline.h>
#include <cstdlib>
#include <cstdio>
#include <curand_kernel.h>
#include <string.h>
#include "DeviceKernel.cu"

extern "C"
{

void cudaInit(int argc, char **argv)
{   
     
}

void allocateArray(void **devPtr, size_t size)
{
    cutilSafeCall(cudaMalloc(devPtr, size));
}

void allocateMatrix(int *pointPtr, size_t pitch, size_t size, int height){

	cudaMallocPitch((void**)&pointPtr, &pitch, size, height);

}

void freeArray(void *devPtr)
{
    cutilSafeCall(cudaFree(devPtr));
}

void threadSync()
{
    cutilSafeCall(cudaThreadSynchronize());
}

void copyArrayFromDevice(void* host, const void* device, int size)
{   
    cutilSafeCall(cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost));
}

void copyArrayToDevice(void* device, const void* host, int offset, int size)
{
   cutilSafeCall(cudaMemcpy(device, host, size, cudaMemcpyHostToDevice));
}

void setParameters(SimParams *hostParams)
{
    // copy parameters to constant memory
    cutilSafeCall( cudaMemcpyToSymbol(params, hostParams, sizeof(SimParams)) );
}

void setkernel(curandState *devStates, int numThreads, int numBlocks){

	dim3 DimGrid(numBlocks,1);
	dim3 DimBlock(numThreads,1,1);

	setup_kernel<<< DimGrid, DimBlock>>>(devStates);

}

void HostCallGo(uint numThreads, uint numBlocks){

}

void HostCallPrintX(uint numThreads, uint numBlocks, curandState* devStates, int* InternalClock,double* ThetaPersist,int* InternalDeviation,float* 
DTimeCrime,double*  FPosition0, double* FPosition1,  double* PPosition0,double*  PPosition1,double* FVelocity0, double* FVelocity1, double* PVelocity0, 
double* PVelocity1,double* FMig0,double* FMig1,double* BIJ0,double* BIJ1,float* DTimePrime,double* DAngRotate,float* DTimePrimeStep,float* 
DTimeCrimeStep,double* Bgamma,long int* TimeInContact,long int* BTimeContact,long int* BTimeInContact,float* gamma,int* AllContacts,double* 
TimeArray, double* 
Bphi,int* Eightypercent, float* PrintOut, int* ZV, double* VelArray){

		dim3 DimGrid(numBlocks,1);
		dim3 DimBlock(numThreads,1,1);
		//Call the Kernel function and start GPU simulation.
		PrintXCoords<<< DimGrid, DimBlock>>>(devStates, InternalClock, ThetaPersist,  InternalDeviation, DTimeCrime,  FPosition0, FPosition1, 
PPosition0,  PPosition1, FVelocity0, FVelocity1, PVelocity0, PVelocity1, FMig0, FMig1, BIJ0, BIJ1, DTimePrime, DAngRotate, DTimePrimeStep, DTimeCrimeStep, 
Bgamma, TimeInContact, BTimeContact, BTimeInContact, gamma, AllContacts, TimeArray, Bphi, Eightypercent, PrintOut, ZV, VelArray);

}
	

void HostCallPrintX1(uint numThreads, uint numBlocks, curandState* devStates, int* InternalClock,double* ThetaPersist,int* InternalDeviation,float* 
						DTimeCrime,double*  FPosition0, double* FPosition1,  double* PPosition0,double*  PPosition1,double* FVelocity0, double* FVelocity1, double* PVelocity0, 
						double* PVelocity1,double* FMig0,double* FMig1,double* BIJ0,double* BIJ1,float* DTimePrime,double* DAngRotate,float* DTimePrimeStep,float* 
						DTimeCrimeStep,double* Bgamma,long int* TimeInContact,long int* BTimeContact,long int* BTimeInContact,float* gamma,int* AllContacts,double* TimeArray, double* Bphi,int* Eightypercent, float* PrintOut, int* ZV, double* VelArray){
		
		dim3 DimGrid(numBlocks,1);
		dim3 DimBlock(numThreads,1,1);
		//Call the Kernel function and start GPU simulation.
		PrintXCoordsOne<<< DimGrid, DimBlock>>>(devStates, InternalClock, ThetaPersist,  InternalDeviation, DTimeCrime,  FPosition0, FPosition1, 
											 PPosition0,  PPosition1, FVelocity0, FVelocity1, PVelocity0, PVelocity1, FMig0, FMig1, BIJ0, BIJ1, DTimePrime, DAngRotate, DTimePrimeStep, DTimeCrimeStep, 
											 Bgamma, TimeInContact, BTimeContact, BTimeInContact, gamma, AllContacts, TimeArray, Bphi, Eightypercent, PrintOut, ZV, VelArray);
		
}


};  

