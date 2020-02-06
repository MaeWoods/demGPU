/*
 *  DEMCode.cuh
 *  
 *  Created by Mae Woods on 19/01/2012.
 *  Licensed under the GNU GPL. This file is free to
 *  use or modify.
 *  Created as part of a PhD project in computational
 *  biology in the department of CoMPLEX UCL
 *
 */

#ifndef POLARITYCODE_H
#define POLARITYCODE_H
#include <curand_kernel.h>
class ModelStandard{

	private:

	public: 
	__device__ ModelStandard(void);
	__device__ ~ModelStandard(void);

	__device__ double* setAll(double StepBy,double AvVt,double Vsumtot,double* phiB,double randomDev,curandState *devStates_D,int* InternalClock,double* 
ThetaPersist,int*  InternalDeviation,float* DTimeC,double* FPosition0, double* FPosition1,  double* PPosition0,double*  
PPosition1,double* FVelocity0, double* FVelocity1, double* PVelocity0, double* PVelocity1,double* FMig0,double* FMig1,double* 
BIJ0,double* BIJ1,float* DTimePrime,double* DAngRotate,float* DTimePrimeStep,float* DTimeCStep,double* Bgamma,long int* TimeInContact,long int* 
BTimeContact,long int* 
BTimeInContact,float* gamma,int* AllContacts,double* TimeArray,int* Eightypercent, float* PrintOut, int* ZV, double* VelArray);
	
__device__ double* setAll1(double StepBy,double AvVt,double Vsumtot,double* phiB, double randomDev,curandState *state,int* InternalClock,double* 
											  ThetaPersist,int* InternalDeviation,float* DTimeC,double*  FPosition0, 
double* FPosition1,double* PPosition0,double* PPosition1,double* FVelocity0, double* FVelocity1,double* PVelocity0, double* PVelocity1,double* FMig0,double* 
FMig1,double* BIJ0,double* BIJ1,float* DTimeP,double* DAngRotate,float* DTimePStep,float* DTimeCStep,double* Bgamma,long int* TimeInContact,long int* 
BTimeContact,long int* BTimeInContact,float* gamma,int* AllContacts,double* TimeArray,int* Eightypercent, float* PrintOut, int* ZV, double* VelArray);

	__device__ void ReturnTimeArray();
	
};

#endif
