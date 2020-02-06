/*
 *  DeviceKernel.cu
 *  
 *  Device code of Migratory DEM created by Mae Woods on 19/01/2012.
 *  Licensed under the GNU GPL. This file is free to
 *  use or modify.
 *  Created as part of a PhD project in computational
 *  biology in the department of CoMPLEX UCL
 *
 */

#ifndef _DEVICE_KERNEL_H_
#define _DEVICE_KERNEL_H_

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <cuda.h>
#include <string.h>
#include <cuPrintf.cu>
#include <cuPrintf.cuh>
#include "math_constants.h"
#include "DeviceKernel.cuh"
#include <curand_kernel.h>
#include <PolarityCode.cuh>
#include <PolarityCode.cu>
#include "device_functions.h"

#define CUPRINTF(fmt, ...) printf(fmt,	__VA_ARGS__)

__constant__ SimParams params;

__global__ void setup_kernel(curandState *state){
	
	double valB = ceilf((blockIdx.x/10)); 
	double param = valB - ((valB-1)*10);
	
	int bx = blockIdx.x;
	int tx = threadIdx.x;

	int id = (bx*50) +tx;

	curand_init(1234-param,id,0,&state[id]);

}

__global__ void PrintXCoords(curandState *devStates_D, int* InternalClock,double* ThetaPersist,int*  InternalDeviation,float* 
CtTime,double*  FPosition0, double* FPosition1,  double* PPosition0,double*  PPosition1,double* FVelocity0, double* FVelocity1, 
double* PVelocity0, double* PVelocity1,double* FMig0,double* FMig1,double* BIJ0,double* BIJ1,float* DtTime,double* 
DAngRotate,float* DtTimeStep,float* CtTimeStep,double* Bgamma,long int* TimeInContact,long int* 
BTimeContact,long int* BTimeInContact,float* gamma,int* AllContacts,double* TimeArray, double* phiB,int* Eightypercent, float* 
PrintOut, int* ZV, double* VelArray){
	
	ModelStandard Ms;

	double valBbegin = ceil(((double(blockIdx.x)+20+0.1)/10.0));
		
		//If not performing a parameter sweep, set different values of the diffusion length 
		//to different block IDs
		if(valBbegin==1){
			
			phiB[blockIdx.x] = 630.134;
			
			
		}
		else if(valBbegin==2){
			
			phiB[blockIdx.x] = 1260.268;
			
		}
		
		else if(valBbegin==3){
			
			phiB[blockIdx.x] = 6301.338;
			
		}
		
		else if(valBbegin==4){
			
			phiB[blockIdx.x] = 12602.676;
			
		}
		
		else if(valBbegin==5){
			
			phiB[blockIdx.x] = 63013.38;
			
		}
		
		else if(valBbegin==6){
			
			phiB[blockIdx.x] = 126026.76;
			
		}
		
		else if(valBbegin==7){
			
			phiB[blockIdx.x] = 630133.801;
			
		}
		
		else if(valBbegin==8){
			
			phiB[blockIdx.x] = 1260267.601;
			
		}
		
		else if(valBbegin==9){
			
			phiB[blockIdx.x] = 6301338.01;
			
		}
		
		else if(valBbegin==10){
			
			phiB[blockIdx.x] = 12602676.01;
			
		}
		else{
			
			phiB[blockIdx.x] = 630.134;
			
		}

	TimeArray = 
Ms.setAll(/*StepBy*/12.0e+7,/*AvVt*/0,/*Vsumtot*/0,phiB,/*randomDev*/360, 
devStates_D, InternalClock, ThetaPersist,  InternalDeviation, CtTime,  FPosition0,  FPosition1,  PPosition0,  PPosition1, FVelocity0,  FVelocity1,  
PVelocity0,  
PVelocity1,FMig0,FMig1,BIJ0,BIJ1,DtTime,DAngRotate,DtTimeStep,CtTimeStep,Bgamma,TimeInContact,BTimeContact,BTimeInContact,gamma,AllContacts,TimeArray,Eightypercent,PrintOut, 
ZV, VelArray);
	
	__syncthreads();

}

__global__ void PrintXCoordsOne(curandState *devStates_D, int* InternalClock,double* ThetaPersist,int*  InternalDeviation,float* 
							 CtTime,double*  FPosition0, double* FPosition1,  double* PPosition0,double*  PPosition1,double* FVelocity0, double* FVelocity1, 
							 double* PVelocity0, double* PVelocity1,double* FMig0,double* FMig1,double* BIJ0,double* BIJ1,float* DtTime,double* 
							 DAngRotate,float* DtTimeStep,float* CtTimeStep,double* Bgamma,long int* TimeInContact,long int* 
							 BTimeContact,long int* BTimeInContact,float* gamma,int* AllContacts,double* TimeArray, double* phiB,int* Eightypercent, float* PrintOut, int* ZV, double* VelArray){
	
	ModelStandard Ms;
	
	double valBbegin = ceil(((double(blockIdx.x)+20+0.1)/10.0));
	
	if(valBbegin==1){
		
		phiB[blockIdx.x] = 630.134;
		
		
	}
	else if(valBbegin==2){
		
		phiB[blockIdx.x] = 1260.268;
		
	}
	
	else if(valBbegin==3){
		
		phiB[blockIdx.x] = 6301.338;
		
	}
	
	else if(valBbegin==4){
		
		phiB[blockIdx.x] = 12602.676;
		
	}
	
	else if(valBbegin==5){
		
		phiB[blockIdx.x] = 63013.38;
		
	}
	
	else if(valBbegin==6){
		
		phiB[blockIdx.x] = 126026.76;
		
	}
	
	else if(valBbegin==7){
		
		phiB[blockIdx.x] = 630133.801;
		
	}
	
	else if(valBbegin==8){
		
		phiB[blockIdx.x] = 1260267.601;
		
	}
	
	else if(valBbegin==9){
		
		phiB[blockIdx.x] = 6301338.01;
		
	}
	
	else if(valBbegin==10){
		
		phiB[blockIdx.x] = 12602676.01;
		
	}
	else{
		
		phiB[blockIdx.x] = 630.134;
		
	}
	
	TimeArray = 
	Ms.setAll1(/*StepBy*/12.0e+7,/*AvVt*/0,/*Vsumtot*/0,phiB,/*randomDev*/360, 
			  devStates_D, InternalClock, ThetaPersist,  InternalDeviation, CtTime,  FPosition0,  FPosition1,  PPosition0,  PPosition1, FVelocity0,  FVelocity1,  
			  PVelocity0,  
			  PVelocity1,FMig0,FMig1,BIJ0,BIJ1,DtTime,DAngRotate,DtTimeStep,CtTimeStep,Bgamma,TimeInContact,BTimeContact,BTimeInContact,gamma,AllContacts,TimeArray,Eightypercent,PrintOut, 
			  ZV, VelArray);
	
	__syncthreads();
	
}


#endif

