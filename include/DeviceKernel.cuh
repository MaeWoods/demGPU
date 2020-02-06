/*
 *  DeviceKernel.cuh
 *  
 *  Created by Mae Woods on 19/01/2012.
 *  Licensed under the GNU GPL. This file is free to
 *  use or modify.
 *  Created as part of a PhD project in computational
 *  biology in the department of CoMPLEX UCL
 *
 */

#ifndef PARTICLES_KERNEL_H
#define PARTICLES_KERNEL_H

#include "vector_types.h"
#include "curand_kernel.h"
typedef unsigned int uint;

#define P_INFORMED 1.0
#define P_UNINFORMED 0.0

extern "C" 
{

__global__ void PrintXCoords(curandState *devStates_D, int* InternalClock, double* ThetaPersist,  int* InternalDeviation, float* 
DTimeCrime,  double* FPosition0,  double* FPosition1,  double* PPosition0,  double* PPosition1, double* FVelocity0,  double* 
FVelocity1,  double* PVelocity0,  double* PVelocity1,double* FMig0,double* FMig1,double* BIJ0,double* BIJ1,float* DTimePrime,double* 
DAngRotate,float* DTimePrimeStep,float* DTimeCrimeStep,double* Bgamma,long int* TimeInContact,long int* BTimeContact,long int* BTimeInContact,float* gamma,int* 
AllContacts, double* TimeArray, double* phiB,int* Eightypercent, float* PrintOut, int* ZV, double* 
VelArray);
	
__global__ void PrintXCoordsOne(curandState *devStates_D, int* InternalClock, double* ThetaPersist,  int* InternalDeviation, float* 
								 DTimeCrime,  double* FPosition0,  double* FPosition1,  double* PPosition0,  double* PPosition1, double* FVelocity0,  double* 
								 FVelocity1,  double* PVelocity0,  double* PVelocity1,double* FMig0,double* FMig1,double* BIJ0,double* BIJ1,float* DTimePrime,double* 
								 DAngRotate,float* DTimePrimeStep,float* DTimeCrimeStep,double* Bgamma,long int* 
TimeInContact,long int* BTimeContact,long int* BTimeInContact,float* gamma,int* AllContacts, double* TimeArray, double* phiB,int* Eightypercent, float* 
PrintOut, 
int* ZV, double* VelArray);

};

// simulation parameters
struct SimParams {
   
    uint gridSize;
    uint numCells;
    double worldOrigin;
    double worldSize;
    double cellSize;
    double gradStrength;

    uint flowGridSize;
    uint fieldGridSize;
    double deltaTime;
    uint numInformed;

    double memory;
    int numThreads;
    int numBlocks;

};

#endif
