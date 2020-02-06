/*
 *  DEMSetup.cu
 *  
 *  Created by Mae Woods and Chris Barnes on 01/11/2011.
 *  Licensed under the GNU GPL. This file is free to use or modify.
 *  Department of CoMPLEX UCL
 *
 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string.h>
#include <cuda.h>
#include <cuPrintf.cu>
#include <math.h>
#include <stdio.h>  
#include <cstring>  
#include <cuPrintf.cuh>
#include "cutil_math.h"
#include "math_constants.h"
#include "device_functions.h"
#include "DEMCode.cuh"
#include "DeviceKernel.cuh"

#define THREAD 50
#define PI 3.14159265

const float QBy = 20;
const double TotalTime = 20000;
const double TotalTimeStep =0;
const double DTime = 1.0e-3;
const double ContactDampingSF = 1.2;
const double kNormal = 0.1124365;
const double phiA = 1.0;
const double Radius = 20.0e-6;
const double Mass = 1.0e-10;
const double BoxXMax = 217.0e-6;
const double BoxXMin = 0;
const double BoxYMax = 1600.0e-6;
const double BoxYMin = 0;
const double alphavt = 1.0;
const float LowerBoundLength = 750.0e-6;
const double CSpeed = 5.0e-8;
const int Np = 50; 
const int MaxInternalClock = 4;
const int MaxInternalDeviation = 3; 

__device__ ModelStandard::ModelStandard(){

	
}

__device__ ModelStandard::~ModelStandard(){
	
}	


__device__ double* ModelStandard::setAll(double StepBy,double AvVt,double Vsumtot,double* phiB, 
double randomDev,curandState *state,int* InternalClock,double* 
ThetaPersist,int* InternalDeviation,float* DTimeC,double*  FPosition0, double* FPosition1,double* PPosition0,double* PPosition1,double* FVelocity0, double* FVelocity1,double* PVelocity0, double* PVelocity1,double* FMig0,double* FMig1,double* BIJ0,double* BIJ1,float* DTimeP,double* DAngRotate,float* DTimePStep,float* DTimeCStep,double* Bgamma,long int* TimeInContact,long int* BTimeContact,long int* 
BTimeInContact,float* gamma,int* AllContacts,double* TimeArray,int* Eightypercent, float* PrintOut, int* ZV, double* VelArray){	
	
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	
	int idvec = (bx * Np) + tx;
	curandState functionState= state[idvec];
	
	int MB = Np*Np*bx;
	
	///////Allocate all matrices in shared memory///////////
	__shared__ double B_0[THREAD];
	__shared__ int All_Cont[THREAD];
	__shared__ double B_1[THREAD];
	__shared__ double Bg_a[THREAD];
	__shared__ long int BT_Contact[THREAD];
	__shared__ long int BT_InContact[THREAD];
	__shared__ int ZeroVel[THREAD];
	
	__shared__ double P0[THREAD];
	__shared__ double P1[THREAD];
	__shared__ double FP0[THREAD];
	__shared__ double FP1[THREAD];
	__shared__ double PV0[THREAD];
	__shared__ double PV1[THREAD];
	__shared__ double FV0[THREAD];
	__shared__ double FV1[THREAD];
	__shared__ int I_D[THREAD];
	__shared__ int I_C[THREAD];
	__shared__ double DR_rot[THREAD];
	__shared__ float d_tC[THREAD];
	__shared__ double F0[THREAD];
	__shared__ double F1[THREAD];
	__shared__ float d_tP[THREAD];
	__shared__ float d_tPstep[THREAD];
	__shared__ float d_tCstep[THREAD];
	__shared__ double Theta_P[THREAD];
	__shared__ long int TimeContact[THREAD*THREAD];
	__shared__ float g_a[THREAD*THREAD];

	__shared__ double Vxx;
	__shared__ double Vyy;
	__shared__ double SumVxx;
	__shared__ double SumVyy;
	__shared__ double ModV;
	__shared__ double CounterV;
	__shared__ double returnvalueV;	
	__shared__ int NumCells;

	Vxx = 0;
	Vyy = 0;
	SumVxx = 0;
	SumVyy = 0;
	ModV = 0;
	CounterV = 0;
	returnvalueV = 0;
	NumCells = 0;
	
	long int CounterP = 0;	
	int TimeDatArr = 0;
	
		double nr = curand_uniform(&functionState);
        ThetaPersist[idvec] = nr*2*PI;
        state[idvec]=functionState;

        double origrand = curand_uniform(&functionState);
        InternalDeviation[idvec] = int(ceil((origrand)*double(MaxInternalDeviation)))-1;
        state[idvec]=functionState;
        double newrand = curand_uniform(&functionState);
        InternalClock[idvec] = int(ceil((newrand)*double(MaxInternalClock)))-1;
        state[idvec]=functionState;

        double newrand2 = curand_uniform(&functionState);
        DTimeC[idvec] = (newrand2)*0.1;
        state[idvec]=functionState;
        AllContacts[idvec]=0;
        BTimeContact[idvec] = 0;
        BTimeInContact[idvec] = 0;
        Bgamma[idvec] = 0;
        double newrand3 = curand_uniform(&functionState);
        DTimeP[idvec] = 150*newrand3;
        state[idvec]=functionState;
        double newrand4 = curand_uniform(&functionState);
        DTimePStep[idvec] = 0.1*newrand4;
	    state[idvec]=functionState;
        double newrand5 = curand_uniform(&functionState);
        DTimeCStep[idvec] = 0;
        state[idvec]=functionState;
		FMig0[idvec] = 0;
		FMig1[idvec] = 0;
		DAngRotate[idvec] = 0;
		FPosition0[idvec] = 0;
		FPosition1[idvec] = 0;
		BIJ0[idvec] = 0;
		BIJ1[idvec] = 0;
		FMig0[idvec] = 0;
		FMig1[idvec] = 0;
		ZV[idvec] = 0;
	
	for(int j=0; j<Np; j++){
		
		TimeInContact[MB + j*Np + tx]=0;
		gamma[MB + j*Np + tx]=0;
		
	}
	
	////////Load the matrices from global memory to shared memory////////
	
	I_D[tx] = InternalDeviation[idvec];
	I_C[tx] = InternalClock[idvec];
	d_tC[tx] = DTimeC[idvec];
	P0[tx]=PPosition0[idvec];
	P1[tx]=PPosition1[idvec];
	FP0[tx]=FPosition0[idvec];
	FP1[tx]=FPosition1[idvec];
	PV0[tx]=PVelocity0[idvec];
	PV1[tx]=PVelocity1[idvec];
	FV0[tx]=FVelocity0[idvec];
	FV1[tx]=FVelocity1[idvec];
	F0[tx] = FMig0[idvec];
	F1[tx] = FMig1[idvec];
	d_tP[tx] = DTimeP[idvec];
	d_tPstep[tx] = DTimePStep[idvec];
	d_tCstep[tx] = DTimeCStep[idvec];
	Theta_P[tx] = ThetaPersist[idvec];
	DR_rot[tx] = DAngRotate[idvec];
	B_0[tx] = BIJ0[idvec];
	B_1[tx] = BIJ1[idvec];
	Bg_a[tx] = Bgamma[idvec];
	BT_Contact[tx] = BTimeContact[idvec];
	BT_InContact[tx] = BTimeInContact[idvec];
	All_Cont[tx] = AllContacts[idvec];
	ZeroVel[tx] = ZV[idvec];
	
	for(int j=0; j<Np; j++){
		
		TimeContact[j*Np + tx] = TimeInContact[MB + j*Np + tx];
		g_a[j*Np + tx] = gamma[MB + j*Np + tx];
		
	}
	
	//Initialize positions of cells
	
	P0[tx] = BoxXMin + 5.0e-6 + ((1.0e-6)) + Radius + (double(tx) - 5*(double(ceil((double(tx)+0.01)/5))-1))*(2*Radius + 1e-6);
	P1[tx] = BoxYMax + 3*Radius - (1e-6) + (2*Radius + 1e-6)*(-1*ceil((double(tx)+1)/5)-1);
	
	//Initialize forces and velocity
	
	ZeroVel[tx] = 0;	
	PV0[tx]=0;
	PV1[tx]=0;
	FV0[tx]=0;
	FV1[tx]=0;
	F0[tx]=0;
	F1[tx]=0;
	B_0[tx] = 0;
	B_1[tx] = 0;
	
	if(bx==0){
		
		//Check things are working with cuPrintf
		cuPrintf("BlockIdx: %d BPhi: %f TP0: %f TP1: %f TV0: %4.10f TV1: %4.10f ThetaP %4.10f \n",bx,1.2,P0[tx],P1[tx],F0[tx],F1[tx],Theta_P[tx]);
		
	}
	
	__syncthreads();

	for(double Time=0; Time<TotalTime; Time += DTime){
		
		//Set the polarity of the cell
		if(Time>0){
			
			Theta_P[tx] = 0;
			Theta_P[tx] = atan2(PV1[tx],PV0[tx]);
			
		}
		
		//Increment the internal clock by 1, or reset
		if(d_tCstep[tx]==0.0){
			
			int MIC = (I_C[tx] >= MaxInternalClock) ? 0 : 1;
			I_C[tx] = MIC*(I_C[tx]+1);
			
		}
		
		//Increment the internal deviation by 1, or reset
		if(d_tPstep[tx]==0.0){
			
			int MID = (I_D[tx] >= MaxInternalDeviation) ? 0 : 1;
			I_D[tx] = MID*(I_D[tx]+1);
			
		}
		
		///Boundary Conditions///
		
		
		double BL, BY, BVX, BVY, BIJMag, BOverlap, BVelRn, BcNormal, Btheta, newangle;
		
		if(P0[tx]>BoxXMax-Radius){
			
			BL = 0;
			BY = 0;
			BVX = 0;
			BVY = 0;
			BIJMag = 0;
			BOverlap = 0;
			BVelRn = 0;
			BcNormal = 0;
			Btheta = 0;
			newangle = 0;
            
			newangle = atan2(PV1[tx],(-1*PV0[tx]));
			
			BL = BoxXMax + Radius;
			BY = P1[tx];
			BVX = 0;
			BVY = 0;
			
			B_0[tx] = BL - P0[tx];
			B_1[tx] = BY - P1[tx];
			BIJMag = sqrt(((B_0[tx]*B_0[tx]))+((B_1[tx]*B_1[tx])));
			BOverlap = sqrt(((B_0[tx]*B_0[tx]))+((B_1[tx]*B_1[tx]))) - Radius - Radius;
			
			if(BOverlap<0){
				
				All_Cont[tx] = 1;
				
				F0[tx] -= kNormal*sqrt(fabs(BOverlap*BOverlap*BOverlap))*(B_0[tx]/BIJMag);
				F1[tx] -= kNormal*sqrt(fabs(BOverlap*BOverlap*BOverlap))*(B_1[tx]/BIJMag);
				
				BVelRn = (PV0[tx]-BVX)*(B_0[tx]/BIJMag) + (PV1[tx]-BVY)*(B_1[tx]/BIJMag);
				BcNormal = ContactDampingSF*sqrt(((Mass/2.0)*kNormal*sqrt(fabs(BOverlap))));
				
				F0[tx] -=  BcNormal*BVelRn*(B_0[tx]/BIJMag);
				F1[tx] -=  BcNormal*BVelRn*(B_1[tx]/BIJMag);
				
				if(BT_Contact[tx]==0){
					
					BT_Contact[tx] = 1;
					
				}
				
				if(BT_Contact[tx]==1){
					double r = curand_uniform(&functionState);
					//theta = atan2((B_1[tx]/BIJMag),(B_0[tx]/BIJMag));
					state[idvec]=functionState;		
					
					Bg_a[tx] = ((r-0.5)*180*(PI/180));
					BT_Contact[tx]=2;					
				}
				
				double theta = atan2((B_1[tx]/BIJMag),(B_0[tx]/BIJMag));
				BT_InContact[tx] += 1;
				double passedtime = (BT_InContact[tx] > 600000) ? 1 : 0;
				
				F0[tx] -= passedtime*Mass*alphavt*(CSpeed*cos(theta + Bg_a[tx]) - PV0[tx]);
				F1[tx] -= passedtime*Mass*alphavt*(CSpeed*sin(theta + Bg_a[tx]) - PV1[tx]);
				
				I_C[tx]	= (MaxInternalClock-1);
				double newrand = curand_uniform(&functionState);
				d_tC[tx] = newrand*100;
				state[idvec]=functionState;
				d_tCstep[tx]=2;
			}
			
			else{
				
				BT_InContact[tx] = 0;
				BT_Contact[tx]=0;
				Bg_a[tx] = 0;
				
				
			}
			
		}
		
		if(P0[tx]<BoxXMin+Radius){
			
			BL = 0;
			BY = 0;
			BVX = 0;
			BVY = 0;
			BIJMag = 0;
			BOverlap = 0;
			BVelRn = 0;
			BcNormal = 0;
			Btheta = 0;
			newangle = 0;
			
            newangle = atan2(PV1[tx],(-1*PV0[tx]));
            
			BL = BoxXMin - Radius;
			BY = P1[tx];
			BVX = 0;
			BVY = 0;
			
			B_0[tx] = BL - P0[tx];
			B_1[tx] = BY - P1[tx];
			BIJMag = sqrt(((B_0[tx]*B_0[tx]))+((B_1[tx]*B_1[tx])));
			BOverlap = sqrt(((B_0[tx]*B_0[tx]))+((B_1[tx]*B_1[tx]))) - 2*Radius;
			
			
			if(BOverlap<0){
				
				All_Cont[tx] = 1;
				
				F0[tx] -= kNormal*sqrt(fabs(BOverlap*BOverlap*BOverlap))*(B_0[tx]/BIJMag);
				F1[tx] -= kNormal*sqrt(fabs(BOverlap*BOverlap*BOverlap))*(B_1[tx]/BIJMag);
				
				BVelRn = (PV0[tx]-BVX)*(B_0[tx]/BIJMag) + (PV1[tx]-BVY)*(B_1[tx]/BIJMag);
				BcNormal = ContactDampingSF*sqrt(((Mass/2.0)*kNormal*sqrt(fabs(BOverlap))));
				
				F0[tx] -=  BcNormal*BVelRn*(B_0[tx]/BIJMag);
				F1[tx] -=  BcNormal*BVelRn*(B_1[tx]/BIJMag);
				
				if(BT_Contact[tx]==0){
					
					BT_Contact[tx] = 1;
					
					
				}
				
				if(BT_Contact[tx]==1){
					double r = curand_uniform(&functionState);
					//theta = atan2((B_1[tx]/BIJMag),(B_0[tx]/BIJMag));
					state[idvec]=functionState;		
					
					Bg_a[tx] = ((r-0.5)*180*(PI/180));
					BT_Contact[tx]=2;					
				}
				
				double theta = atan2((B_1[tx]/BIJMag),(B_0[tx]/BIJMag));
				BT_InContact[tx] += 1;
				double passedtime = (BT_InContact[tx] > 600000) ? 1 : 0;
				
				F0[tx] -= passedtime*Mass*alphavt*(CSpeed*cos(theta + Bg_a[tx])-PV0[tx]);
				F1[tx] -= passedtime*Mass*alphavt*(CSpeed*sin(theta + Bg_a[tx])-PV1[tx]);
				
				I_C[tx]	= (MaxInternalClock-1);
				double newrand = curand_uniform(&functionState);
				d_tC[tx] = newrand*100;
				state[idvec]=functionState;
				d_tCstep[tx]=2;
			}
			
			else{
				
				BT_InContact[tx] = 0;
				BT_Contact[tx]=0;
				Bg_a[tx] = 0;
				
				
			}
			
		}
		
		if(P1[tx]>BoxYMax-Radius){
			
			BL = 0;
			BY = 0;
			BVX = 0;
			BVY = 0;
			BIJMag = 0;
			BOverlap = 0;
			BVelRn = 0;
			BcNormal = 0;
			Btheta = 0;
			newangle = 0;
			
            newangle = atan2((-1*PV1[tx]),PV0[tx]);
            
			BL = P0[tx];
			BY = BoxYMax + Radius;
			BVX = 0;
			BVY = 0;
			
			B_0[tx] = BL - P0[tx];
			B_1[tx] = BY - P1[tx];
			BIJMag = sqrt(((B_0[tx]*B_0[tx]))+((B_1[tx]*B_1[tx])));
			BOverlap = sqrt(((B_0[tx]*B_0[tx]))+((B_1[tx]*B_1[tx]))) - 2*Radius;
			
			if(BOverlap<0){
				
				All_Cont[tx] = 1;
				
				F0[tx] -= kNormal*sqrt(fabs(BOverlap*BOverlap*BOverlap))*(B_0[tx]/BIJMag);
				F1[tx] -= kNormal*sqrt(fabs(BOverlap*BOverlap*BOverlap))*(B_1[tx]/BIJMag);
				
				BVelRn = (PV0[tx]-BVX)*(B_0[tx]/BIJMag) + (PV1[tx]-BVY)*(B_1[tx]/BIJMag);
				BcNormal = ContactDampingSF*sqrt(((Mass/2.0)*kNormal*sqrt(fabs(BOverlap))));
				
				F0[tx] -=  BcNormal*BVelRn*(B_0[tx]/BIJMag);
				F1[tx] -=  BcNormal*BVelRn*(B_1[tx]/BIJMag);
				
				if(BT_Contact[tx]==0){
					BT_Contact[tx] = 1;
					
					
				}
				if(BT_Contact[tx]==1){
					double r = curand_uniform(&functionState);
					//theta = atan2((B_1[tx]/BIJMag),(B_0[tx]/BIJMag));
					state[idvec]=functionState;		
					
					Bg_a[tx] = ((r-0.5)*180*(PI/180));
					BT_Contact[tx]=2;					
				}
				
				double theta = atan2((B_1[tx]/BIJMag),(B_0[tx]/BIJMag));
				BT_InContact[tx] += 1;
				double passedtime = (BT_InContact[tx] > 600000) ? 1 : 0;
				
				F0[tx] -= passedtime*Mass*alphavt*(CSpeed*cos(theta + Bg_a[tx])-PV0[tx]);
				F1[tx] -= passedtime*Mass*alphavt*(CSpeed*sin(theta + Bg_a[tx])-PV1[tx]);
				
				I_C[tx]	= (MaxInternalClock-1);
				double newrand = curand_uniform(&functionState);
				d_tC[tx] = newrand*100;
				state[idvec]=functionState;
				d_tCstep[tx]=2;
			}
			
			else{
				
				BT_InContact[tx] = 0;
				BT_Contact[tx]=0;
				Bg_a[tx] = 0;
				
				
			}
			
		}
		
		// ==========================================================================
		// Sensing the co-attractant and finding contact
		// ==========================================================================
		
		double RIJMag, Overlap, VelRn, cNormal, PhiXp, PhiXm, PhiYp, PhiYm, PX, PY;
		
		double dPhiX, dPhiY, dPhiMag, dPerX, dPerY, dPerMag, NTP;
		
		PhiXp = 0;
		PhiXm = 0;
		PhiYp = 0;
		PhiYm = 0;
		PX = (Radius-2.0e-6)*cos(Theta_P[tx]);
		PY = (Radius-2.0e-6)*sin(Theta_P[tx]);
		
		// For all cells
		for(int j=0; j<Np; j++){
		
			// ==========================================================================
			// Measure concentration of co-attractant
			// ==========================================================================
			
			PhiXp += phiA*exp( -1*phiB[bx] * sqrt((( (P0[j]-(P0[tx]+PX+1.0e-6))*(P0[j]-(P0[tx]+PX+1.0e-6))))+(( (P1[j]-(P1[tx]+PY))*(P1[j]-(P1[tx]+PY))))));
			PhiXm += phiA*exp( -1*phiB[bx] * sqrt((( (P0[j]-(P0[tx]+PX-1.0e-6))* (P0[j]-(P0[tx]+PX-1.0e-6))))+(( (P1[j]-(P1[tx]+PY))*(P1[j]-(P1[tx]+PY))))));
			
			PhiYp += phiA*exp( -1*phiB[bx] * sqrt((( (P0[j]-(P0[tx]+PX))*(P0[j]-(P0[tx]+PX))))+(( (P1[j]-(P1[tx]+PY+1.0e-6))*(P1[j]-(P1[tx]+PY+1.0e-6))))));
			PhiYm += phiA*exp( -1*phiB[bx] * sqrt((( (P0[j]-(P0[tx]+PX))*(P0[j]-(P0[tx]+PX))))+(( (P1[j]-(P1[tx]+PY-1.0e-6))*(P1[j]-(P1[tx]+PY-1.0e-6))))));
			
			if(tx!=j){
				
				RIJMag = 0;
				Overlap = 0;
				VelRn = 0;
				cNormal = 0;
				
				RIJMag = sqrt((((P0[j]-P0[tx])*(P0[j]-P0[tx])))+(((P1[j]-P1[tx])*(P1[j]-P1[tx]))));
				Overlap = sqrt((((P0[j]-P0[tx])*(P0[j]-P0[tx])))+(((P1[j]-P1[tx])*(P1[j]-P1[tx]))))-Radius-Radius;
				
				// ==========================================================================
				// If contact
				// ==========================================================================
				
				if(Overlap<0){
					
					All_Cont[tx] = 1;
					
					F0[tx] -= kNormal*sqrt(fabs(Overlap*Overlap*Overlap))*((P0[j]-P0[tx])/RIJMag);
					F1[tx] -= kNormal*sqrt(fabs(Overlap*Overlap*Overlap))*((P1[j]-P1[tx])/RIJMag);
					
					VelRn = (PV0[tx]-PV0[j])*((P0[j]-P0[tx])/RIJMag) + (PV1[tx]-PV1[j])*((P1[j]-P1[tx])/RIJMag);
					cNormal = ContactDampingSF*sqrt(((Mass/2.0)*kNormal*sqrt(fabs(Overlap))));
					
					F0[tx] -=  cNormal*VelRn*((P0[j]-P0[tx])/RIJMag);
					F1[tx] -=  cNormal*VelRn*((P1[j]-P1[tx])/RIJMag);
					
					double theta = atan2(((P1[j]-P1[tx])/RIJMag),((P0[j]-P0[tx])/RIJMag));
					TimeContact[j*Np + tx] += 1;
					
					if(TimeContact[j*Np + tx]==1){
						
						float r = curand_uniform(&functionState);
						
						g_a[j*Np + tx] = ((r-0.5)*180*(PI/180));
						state[idvec]=functionState;
					}
					
					double passedtime = (TimeContact[j*Np + tx] > 600000) ? 1 : 0;
					
					F0[tx] -= passedtime*Mass*0.03*(CSpeed*cos(theta + g_a[j*Np + tx]) - PV0[tx]);
					F1[tx] -= passedtime*Mass*0.03*(CSpeed*sin(theta + g_a[j*Np + tx]) - PV1[tx]);
					
					I_C[tx] = (MaxInternalClock-1);
					double newrand = curand_uniform(&functionState);
					d_tC[tx] = newrand*100;
					state[idvec]=functionState;
					d_tCstep[tx]=2;
					
				}

				else{
					
					TimeContact[j*Np + tx] = 0;
					g_a[j*Np + tx] = 0;
					
					
				}
				
			}
			
			
		}
		
		__syncthreads();
		
		//---Checking whether to sense the chemo-attractant at this stage---//
		
		double notclock = (I_C[tx]!=(MaxInternalClock-1)) ? 1 : 0;
		double noCont = (All_Cont[tx]!=1) ? 1 : 0;
		
		dPhiX = 0;
		dPhiY = 0;
		dPhiMag = 0;
		
		dPhiX = PhiXp - PhiXm;
		dPhiY = PhiYp - PhiYm;
		dPhiMag = sqrt((dPhiX*dPhiX)+(dPhiY*dPhiY));
		
		double NewAngAtt = atan2(dPhiY,dPhiX);
		
		F0[tx] += notclock*noCont*Mass*dPhiMag*alphavt*(CSpeed*cos(NewAngAtt)-PV0[tx]);
		F1[tx] += notclock*noCont*Mass*dPhiMag*alphavt*(CSpeed*sin(NewAngAtt)-PV1[tx]);
				
		//---Checking whether to tumble at this stage---//
		
		double MaxDev = (I_D[tx]==(MaxInternalDeviation-1)) ? 1 : 0;
		double MaxClock = (I_C[tx]==(MaxInternalClock-1)) ? 1 : 0;
		
		dPerX = 0;
		dPerY = 0;
		dPerMag = 0;
		NTP = Theta_P[tx] + DR_rot[tx];
		
		dPerX = cos(NTP);
		dPerY = sin(NTP);
		
		F0[tx] += MaxDev*noCont*(Mass/DTime)*((sqrt((PV0[tx]*PV0[tx])+(PV1[tx]*PV1[tx])))*dPerX - PV0[tx]);
		F1[tx] += MaxDev*noCont*(Mass/DTime)*((sqrt((PV0[tx]*PV0[tx])+(PV1[tx]*PV1[tx])))*dPerY - PV1[tx]);
		
		I_D[tx] = I_D[tx]*(1-MaxDev);
		
		//---If no other signals maintain a constant speed---//
		
		double notDev = (I_D[tx]!=(MaxInternalDeviation-1)) ? 1 : 0;
		
		F0[tx] += (1-notclock)*notDev*noCont*Mass*alphavt*(CSpeed*cos(Theta_P[tx])-PV0[tx]);
		F1[tx] += (1-notclock)*notDev*noCont*Mass*alphavt*(CSpeed*sin(Theta_P[tx])-PV1[tx]);
		
		//----Update the internal clocks----//
		
		d_tP[tx] += DTime;
		
		if(((d_tP[tx]>110))||(I_D[tx]==(MaxInternalDeviation-1))){
			
			double ras = curand_uniform(&functionState);
			d_tPstep[tx]=0.0;
			DR_rot[tx] = (ras-0.5)*6;
			state[idvec]=functionState;
			d_tP[tx] = 0;
			
		}
		else{
			
			d_tPstep[tx]=1.0;
			
		}
		
		
		if(d_tCstep[tx]==0){
			
			d_tCstep[tx]=2;
			d_tC[tx] = 0.0;
			
		}
		
		if(I_C[tx]==(MaxInternalClock-1)){
			
			d_tC[tx] += DTime/QBy;
			
			if(d_tC[tx]>0.1){
				
				d_tCstep[tx]=0;
				d_tC[tx] = 0;
			}
			
		}
		
		else{
			
			d_tC[tx] += DTime;
			
			if(d_tC[tx]>0.1){
				
				
				d_tCstep[tx]=0;
				d_tC[tx] = 0;
			}
			
		}
		
		
		//------CENTRAL DIFFERENCE SCHEME-------//
		
		//--Keep cells that have reached the target in their final positions--//
		
		if(P1[tx]<=LowerBoundLength){
			
			P1[tx] = LowerBoundLength-41.4e-6;
			FP1[tx] = LowerBoundLength-41.4e-6;
			
			PV0[tx] = 0;
			PV1[tx] = 0;
			
			FV0[tx] = PV0[tx];
			FV1[tx] = PV1[tx];
			
			FP0[tx] = P0[tx];
			FP1[tx] = P1[tx];
			
			PV0[tx] = FV0[tx];
			PV1[tx] = FV1[tx];
			P0[tx] = FP0[tx];
			P1[tx] = FP1[tx];
			
			ZeroVel[tx]=1;
			
		}
		
		else{
			
			FV0[tx] = PV0[tx] + ((F0[tx])/Mass)*DTime;
			FV1[tx] = PV1[tx] + ((F1[tx])/Mass)*DTime;
			
			FP0[tx] = P0[tx] + FV0[tx]*DTime;
			FP1[tx] = P1[tx] + FV1[tx]*DTime;
			
			P0[tx] = 0;
			P1[tx] = 0;
			PV0[tx] = 0;
			PV1[tx] = 0;			
			P0[tx] = FP0[tx];
			P1[tx] = FP1[tx];
			PV0[tx] = FV0[tx];
			PV1[tx] = FV1[tx];
			
		}
		
		
		//------END CENTRAL DIFFERENCE SCHEME-------//
		
		//----Get simulation results for movie generation----//
		
		if((bx==9)&&(tx==1)){		
			if(CounterP==80000){
				
				
				
				for(int j=0; j<Np; j++){
					
					TimeArray[TimeDatArr*2*Np + j] = P0[j];
					
					TimeArray[TimeDatArr*2*Np + Np + j] = P1[j];
					
					VelArray[TimeDatArr*2*Np + j] = PV0[j];
					
					VelArray[TimeDatArr*2*Np + Np + j] = PV1[j];
					
				}
				
				TimeDatArr += 1;
				
				
				CounterP = 0;
				
			}
			
			CounterP += 1;
			
		}

		
		//---------------------------------------------------//
		
		//----Get 80% at boundary simulation results----//
		
		if(tx==1){
			for(int h=0; h<Np; h++){
				if(P1[h]<LowerBoundLength){
					
					Eightypercent[bx] += 1;
					
				}
			}
			
			if(Eightypercent[bx]>=40){
				
				if((bx==1)){
					
					cuPrintf("Time = %f BlockIdx: %d BPhi: %f TP0: %f TP1: %f TV0: %4.10f TV1: %4.10f \n",Time,bx,phiB[bx],FP0[tx],FP1[tx],FV0[tx],FV1[tx]);
					
				}
				
				PrintOut[bx]=Time;
				
				Eightypercent[bx] = 3*Np;
			}		
			
			
			if(Eightypercent[bx]>=40){
				
				Time = TotalTime+TotalTimeStep;
				
			}
		}
		//-----------------------------------------------//
		
		//Reset vectors
		All_Cont[tx]=0;
		F0[tx] = 0;
		F1[tx] = 0;
		FV0[tx] = 0;
		FV1[tx] = 0;
		B_0[tx] = 0;
		B_1[tx] = 0;
		FP0[tx] = 0;
		FP1[tx] = 0;
		Eightypercent[bx] = 0;
		
		__syncthreads();
		
		///////////////////////////
		/////End of while loop/////
		///////////////////////////
		
	}

	ZV[idvec] = ZeroVel[tx];
	PPosition0[idvec]=P0[tx];
        PPosition1[idvec]=P1[tx];
        PVelocity0[idvec]=PV0[tx];
        PVelocity1[idvec]=PV1[tx];
		DTimeC[idvec]=d_tC[tx];
	InternalDeviation[idvec]=I_D[tx];
	Bgamma[idvec]=Bg_a[tx];
	BTimeContact[idvec]=BT_Contact[tx];
	gamma[idvec]=g_a[tx];
	TimeInContact[idvec]=TimeContact[tx];
	InternalClock[idvec]=I_C[tx];
	DTimeCStep[idvec]=d_tCstep[tx];
	DTimePStep[idvec]=d_tPstep[tx];
	DAngRotate[idvec]=DR_rot[tx];
	DTimeP[idvec]=d_tP[tx];
	
	
	__syncthreads();
	
	return(TimeArray);	
	
}

__device__ double* ModelStandard::setAll1(double StepBy,double AvVt,double Vsumtot,double* phiB, double randomDev,curandState *state,int* InternalClock,double* 
										 ThetaPersist,int* InternalDeviation,float* DTimeC,double*  FPosition0, double* 
FPosition1,double* PPosition0,double* PPosition1,double* FVelocity0, double* FVelocity1,double* PVelocity0, double* PVelocity1,double* FMig0,double* 
FMig1,double* BIJ0,double* BIJ1,float* DTimeP,double* DAngRotate,float* DTimePStep,float* DTimeCStep,double* Bgamma,long int* TimeInContact,long int* 
BTimeContact,long int* 
										 BTimeInContact,float* gamma,int* AllContacts,double* TimeArray,int* Eightypercent, float* PrintOut, int* ZV, double* VelArray){	
	
	
	//Same implementation as setAll(), however data is input from a file.
	//---CUDA IDs---//
	
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int idvec = (bx * Np) + tx;
	curandState functionState= state[idvec];
	int MB = Np*Np*bx;
	
	int CounterP = 1;	
	int TimeDatArr = 1;
	
	///Allocate all matrices in shared memory///
	
	__shared__ double B_0[THREAD];
	__shared__ int All_Cont[THREAD];
	__shared__ double B_1[THREAD];
	__shared__ double Bg_a[THREAD];
	__shared__ long int BT_Contact[THREAD];
	__shared__ long int BT_InContact[THREAD];
	
	__shared__ double P0[THREAD];
	__shared__ double P1[THREAD];
	__shared__ double FP0[THREAD];
	__shared__ double FP1[THREAD];
	__shared__ double PV0[THREAD];
	__shared__ double PV1[THREAD];
	__shared__ double FV0[THREAD];
	__shared__ double FV1[THREAD];
	__shared__ int I_D[THREAD];
	__shared__ int I_C[THREAD];
	__shared__ double DR_rot[THREAD];
	__shared__ float d_tC[THREAD];
	__shared__ double F0[THREAD];
	__shared__ double F1[THREAD];
	__shared__ float d_tP[THREAD];
	__shared__ float d_tPstep[THREAD];
	__shared__ float d_tCstep[THREAD];
	__shared__ double Theta_P[THREAD];
	__shared__ long int TimeContact[THREAD*THREAD];
	__shared__ float g_a[THREAD*THREAD];
	__shared__ int ZeroVel[THREAD];
	__shared__ int EightyCode[THREAD];
	

	__shared__ double TargetHit;

	//--Initialise--//
	
	double nr = curand_uniform(&functionState);
	ThetaPersist[idvec] = nr*2*PI;
	state[idvec]=functionState;

	AllContacts[idvec]=0;
	BTimeContact[idvec] = 0;
	BTimeInContact[idvec] = 0;

	FMig0[idvec] = 0;
	FMig1[idvec] = 0;
	FPosition0[idvec] = 0;
	FPosition1[idvec] = 0;
	BIJ0[idvec] = 0;
	BIJ1[idvec] = 0;
	FMig0[idvec] = 0;
	FMig1[idvec] = 0;
	Eightypercent[idvec] = 0;

	
	//Load the matrices from global memory to shared memory//
	
	I_D[tx] = InternalDeviation[idvec];
	I_C[tx] = InternalClock[idvec];
	d_tC[tx] = DTimeC[idvec];
	P0[tx]=PPosition0[idvec];
	P1[tx]=PPosition1[idvec];
	FP0[tx]=FPosition0[idvec];
	FP1[tx]=FPosition1[idvec];
	PV0[tx]=PVelocity0[idvec];
	PV1[tx]=PVelocity1[idvec];
	FV0[tx]=FVelocity0[idvec];
	FV1[tx]=FVelocity1[idvec];
	F0[tx] = FMig0[idvec];
	F1[tx] = FMig1[idvec];
	d_tP[tx] = DTimeP[idvec];
	d_tPstep[tx] = DTimePStep[idvec];
	d_tCstep[tx] = DTimeCStep[idvec];
	Theta_P[tx] = ThetaPersist[idvec];
	DR_rot[tx] = DAngRotate[idvec];
	B_0[tx] = BIJ0[idvec];
	B_1[tx] = BIJ1[idvec];
	Bg_a[tx] = Bgamma[idvec];
	BT_Contact[tx] = BTimeContact[idvec];
	BT_InContact[tx] = BTimeInContact[idvec];
	All_Cont[tx] = AllContacts[idvec];
	EightyCode[tx] = Eightypercent[idvec];
	
	for(int j=0; j<Np; j++){
		
		TimeContact[j*Np + tx] = TimeInContact[MB + j*Np + tx];
		g_a[j*Np + tx] = gamma[MB + j*Np + tx];
		
	}
	

	
	ZeroVel[tx]=ZV[idvec];	
	//PV0[tx]=0;
	//PV1[tx]=0;
	FV0[tx]=0;
	FV1[tx]=0;
	F0[tx]=0;
	F1[tx]=0;
	B_0[tx] = 0;
	B_1[tx] = 0;
	
	//----Debug statement----//
	
	if(bx==0){
		
		cuPrintf("BlockIdx: %d BPhi: %f TP0: %f TP1: %f TV0: %4.10f TV1: %4.10f ThetaP %4.10f \n",bx,1.2,P0[tx],P1[tx],PV0[tx],PV1[tx],Theta_P[tx]);
		
	}
	
	//---Boundary---//
	
	double BL, BY, BVX, BVY, BIJMag, BOverlap, BVelRn, BcNormal, Btheta, newangle;
	
	//---RIJ and sensing---//
	
	double RIJMag, Overlap, VelRn, cNormal, PhiXp, PhiXm, PhiYp, PhiYm, PX, PY;
	
	double dPhiX, dPhiY, dPhiMag, dPerX, dPerY, dPerMag, NTP;
	
	//---Internal Clock---//
	
	int MIC, MID, notclock, noCont, MaxDev, MaxClock, notDev;
	
	//---Time loop---//
	
	for(double Time=TotalTime; Time<TotalTime+TotalTimeStep; Time += DTime){
		
		//Set the polarity of the cell
		if(Time>0){
			
			Theta_P[tx] = 0;
			Theta_P[tx] = atan2(PV1[tx],PV0[tx]);
			
		}
		
		//Increment the internal clock by 1, or reset
		if(d_tCstep[tx]==0.0){
			
			int MIC = (I_C[tx] >= MaxInternalClock) ? 0 : 1;
			I_C[tx] = MIC*(I_C[tx]+1);
			
		}
		
		//Increment the internal deviation by 1, or reset
		if(d_tPstep[tx]==0.0){
			
			int MID = (I_D[tx] >= MaxInternalDeviation) ? 0 : 1;
			I_D[tx] = MID*(I_D[tx]+1);
			
		}
		
		///Boundary Conditions///
		
		
		double BL, BY, BVX, BVY, BIJMag, BOverlap, BVelRn, BcNormal, Btheta, newangle;
		
		if(P0[tx]>BoxXMax-Radius){
			
			BL = 0;
			BY = 0;
			BVX = 0;
			BVY = 0;
			BIJMag = 0;
			BOverlap = 0;
			BVelRn = 0;
			BcNormal = 0;
			Btheta = 0;
			newangle = 0;
            
			newangle = atan2(PV1[tx],(-1*PV0[tx]));
			
			BL = BoxXMax + Radius;
			BY = P1[tx];
			BVX = 0;
			BVY = 0;
			
			B_0[tx] = BL - P0[tx];
			B_1[tx] = BY - P1[tx];
			BIJMag = sqrt(((B_0[tx]*B_0[tx]))+((B_1[tx]*B_1[tx])));
			BOverlap = sqrt(((B_0[tx]*B_0[tx]))+((B_1[tx]*B_1[tx]))) - Radius - Radius;
			
			if(BOverlap<0){
				
				All_Cont[tx] = 1;
				
				F0[tx] -= kNormal*sqrt(fabs(BOverlap*BOverlap*BOverlap))*(B_0[tx]/BIJMag);
				F1[tx] -= kNormal*sqrt(fabs(BOverlap*BOverlap*BOverlap))*(B_1[tx]/BIJMag);
				
				BVelRn = (PV0[tx]-BVX)*(B_0[tx]/BIJMag) + (PV1[tx]-BVY)*(B_1[tx]/BIJMag);
				BcNormal = ContactDampingSF*sqrt(((Mass/2.0)*kNormal*sqrt(fabs(BOverlap))));
				
				F0[tx] -=  BcNormal*BVelRn*(B_0[tx]/BIJMag);
				F1[tx] -=  BcNormal*BVelRn*(B_1[tx]/BIJMag);
				
				if(BT_Contact[tx]==0){
					
					BT_Contact[tx] = 1;
					
				}
				
				if(BT_Contact[tx]==1){
					double r = curand_uniform(&functionState);
					//theta = atan2((B_1[tx]/BIJMag),(B_0[tx]/BIJMag));
					state[idvec]=functionState;		
					
					Bg_a[tx] = ((r-0.5)*180*(PI/180));
					BT_Contact[tx]=2;					
				}
				
				double theta = atan2((B_1[tx]/BIJMag),(B_0[tx]/BIJMag));
				BT_InContact[tx] += 1;
				double passedtime = (BT_InContact[tx] > 300000) ? 1 : 0;
				
				F0[tx] -= passedtime*Mass*alphavt*(CSpeed*cos(theta + Bg_a[tx]) - PV0[tx]);
				F1[tx] -= passedtime*Mass*alphavt*(CSpeed*sin(theta + Bg_a[tx]) - PV1[tx]);
				
				I_C[tx]	= (MaxInternalClock-1);
				double newrand = curand_uniform(&functionState);
				d_tC[tx] = newrand*100;
				state[idvec]=functionState;
				d_tCstep[tx]=2;
			}
			
			else{
				
				BT_InContact[tx] = 0;
				BT_Contact[tx]=0;
				Bg_a[tx] = 0;
				
				
			}
			
		}
		
		if(P0[tx]<BoxXMin+Radius){
			
			BL = 0;
			BY = 0;
			BVX = 0;
			BVY = 0;
			BIJMag = 0;
			BOverlap = 0;
			BVelRn = 0;
			BcNormal = 0;
			Btheta = 0;
			newangle = 0;
			
            newangle = atan2(PV1[tx],(-1*PV0[tx]));
            
			BL = BoxXMin - Radius;
			BY = P1[tx];
			BVX = 0;
			BVY = 0;
			
			B_0[tx] = BL - P0[tx];
			B_1[tx] = BY - P1[tx];
			BIJMag = sqrt(((B_0[tx]*B_0[tx]))+((B_1[tx]*B_1[tx])));
			BOverlap = sqrt(((B_0[tx]*B_0[tx]))+((B_1[tx]*B_1[tx]))) - 2*Radius;
			
			
			if(BOverlap<0){
				
				All_Cont[tx] = 1;
				
				F0[tx] -= kNormal*sqrt(fabs(BOverlap*BOverlap*BOverlap))*(B_0[tx]/BIJMag);
				F1[tx] -= kNormal*sqrt(fabs(BOverlap*BOverlap*BOverlap))*(B_1[tx]/BIJMag);
				
				BVelRn = (PV0[tx]-BVX)*(B_0[tx]/BIJMag) + (PV1[tx]-BVY)*(B_1[tx]/BIJMag);
				BcNormal = ContactDampingSF*sqrt(((Mass/2.0)*kNormal*sqrt(fabs(BOverlap))));
				
				F0[tx] -=  BcNormal*BVelRn*(B_0[tx]/BIJMag);
				F1[tx] -=  BcNormal*BVelRn*(B_1[tx]/BIJMag);
				
				if(BT_Contact[tx]==0){
					
					BT_Contact[tx] = 1;
					
					
				}
				
				if(BT_Contact[tx]==1){
					double r = curand_uniform(&functionState);
					//theta = atan2((B_1[tx]/BIJMag),(B_0[tx]/BIJMag));
					state[idvec]=functionState;		
					
					Bg_a[tx] = ((r-0.5)*180*(PI/180));
					BT_Contact[tx]=2;					
				}
				
				double theta = atan2((B_1[tx]/BIJMag),(B_0[tx]/BIJMag));
				BT_InContact[tx] += 1;
				double passedtime = (BT_InContact[tx] > 300000) ? 1 : 0;
				
				F0[tx] -= passedtime*Mass*alphavt*(CSpeed*cos(theta + Bg_a[tx])-PV0[tx]);
				F1[tx] -= passedtime*Mass*alphavt*(CSpeed*sin(theta + Bg_a[tx])-PV1[tx]);
				
				I_C[tx]	= (MaxInternalClock-1);
				double newrand = curand_uniform(&functionState);
				d_tC[tx] = newrand*100;
				state[idvec]=functionState;
				d_tCstep[tx]=2;
			}
			
			else{
				
				BT_InContact[tx] = 0;
				BT_Contact[tx]=0;
				Bg_a[tx] = 0;
				
				
			}
			
		}
		
		if(P1[tx]>BoxYMax-Radius){
			
			BL = 0;
			BY = 0;
			BVX = 0;
			BVY = 0;
			BIJMag = 0;
			BOverlap = 0;
			BVelRn = 0;
			BcNormal = 0;
			Btheta = 0;
			newangle = 0;
			
            newangle = atan2((-1*PV1[tx]),PV0[tx]);
            
			BL = P0[tx];
			BY = BoxYMax + Radius;
			BVX = 0;
			BVY = 0;
			
			B_0[tx] = BL - P0[tx];
			B_1[tx] = BY - P1[tx];
			BIJMag = sqrt(((B_0[tx]*B_0[tx]))+((B_1[tx]*B_1[tx])));
			BOverlap = sqrt(((B_0[tx]*B_0[tx]))+((B_1[tx]*B_1[tx]))) - 2*Radius;
			
			if(BOverlap<0){
				
				All_Cont[tx] = 1;
				
				F0[tx] -= kNormal*sqrt(fabs(BOverlap*BOverlap*BOverlap))*(B_0[tx]/BIJMag);
				F1[tx] -= kNormal*sqrt(fabs(BOverlap*BOverlap*BOverlap))*(B_1[tx]/BIJMag);
				
				BVelRn = (PV0[tx]-BVX)*(B_0[tx]/BIJMag) + (PV1[tx]-BVY)*(B_1[tx]/BIJMag);
				BcNormal = ContactDampingSF*sqrt(((Mass/2.0)*kNormal*sqrt(fabs(BOverlap))));
				
				F0[tx] -=  BcNormal*BVelRn*(B_0[tx]/BIJMag);
				F1[tx] -=  BcNormal*BVelRn*(B_1[tx]/BIJMag);
				
				if(BT_Contact[tx]==0){
					BT_Contact[tx] = 1;
					
					
				}
				if(BT_Contact[tx]==1){
					double r = curand_uniform(&functionState);
					//theta = atan2((B_1[tx]/BIJMag),(B_0[tx]/BIJMag));
					state[idvec]=functionState;		
					
					Bg_a[tx] = ((r-0.5)*180*(PI/180));
					BT_Contact[tx]=2;					
				}
				
				double theta = atan2((B_1[tx]/BIJMag),(B_0[tx]/BIJMag));
				BT_InContact[tx] += 1;
				double passedtime = (BT_InContact[tx] > 300000) ? 1 : 0;
				
				F0[tx] -= passedtime*Mass*alphavt*(CSpeed*cos(theta + Bg_a[tx])-PV0[tx]);
				F1[tx] -= passedtime*Mass*alphavt*(CSpeed*sin(theta + Bg_a[tx])-PV1[tx]);
				
				I_C[tx]	= (MaxInternalClock-1);
				double newrand = curand_uniform(&functionState);
				d_tC[tx] = newrand*100;
				state[idvec]=functionState;
				d_tCstep[tx]=2;
			}
			
			else{
				
				BT_InContact[tx] = 0;
				BT_Contact[tx]=0;
				Bg_a[tx] = 0;
				
				
			}
			
		}
		
		double RIJMag, Overlap, VelRn, cNormal, PhiXp, PhiXm, PhiYp, PhiYm, PX, PY;
		
		double dPhiX, dPhiY, dPhiMag, dPerX, dPerY, dPerMag, NTP;
		
		PhiXp = 0;
		PhiXm = 0;
		PhiYp = 0;
		PhiYm = 0;
		PX = (Radius-2.0e-6)*cos(Theta_P[tx]);
		PY = (Radius-2.0e-6)*sin(Theta_P[tx]);
		
		for(int j=0; j<Np; j++){
			
			PhiXp += phiA*exp( -1*phiB[bx] * sqrt((( (P0[j]-(P0[tx]+PX+1.0e-6))*(P0[j]-(P0[tx]+PX+1.0e-6))))+(( (P1[j]-(P1[tx]+PY))*(P1[j]-(P1[tx]+PY))))));
			PhiXm += phiA*exp( -1*phiB[bx] * sqrt((( (P0[j]-(P0[tx]+PX-1.0e-6))* (P0[j]-(P0[tx]+PX-1.0e-6))))+(( (P1[j]-(P1[tx]+PY))*(P1[j]-(P1[tx]+PY))))));
			
			PhiYp += phiA*exp( -1*phiB[bx] * sqrt((( (P0[j]-(P0[tx]+PX))*(P0[j]-(P0[tx]+PX))))+(( (P1[j]-(P1[tx]+PY+1.0e-6))*(P1[j]-(P1[tx]+PY+1.0e-6))))));
			PhiYm += phiA*exp( -1*phiB[bx] * sqrt((( (P0[j]-(P0[tx]+PX))*(P0[j]-(P0[tx]+PX))))+(( (P1[j]-(P1[tx]+PY-1.0e-6))*(P1[j]-(P1[tx]+PY-1.0e-6))))));
			
			if(tx!=j){
				
				RIJMag = 0;
				Overlap = 0;
				VelRn = 0;
				cNormal = 0;
				
				RIJMag = sqrt((((P0[j]-P0[tx])*(P0[j]-P0[tx])))+(((P1[j]-P1[tx])*(P1[j]-P1[tx]))));
				Overlap = sqrt((((P0[j]-P0[tx])*(P0[j]-P0[tx])))+(((P1[j]-P1[tx])*(P1[j]-P1[tx]))))-Radius-Radius;
				
				if(Overlap<0){
					
					All_Cont[tx] = 1;
					
					F0[tx] -= kNormal*sqrt(fabs(Overlap*Overlap*Overlap))*((P0[j]-P0[tx])/RIJMag);
					F1[tx] -= kNormal*sqrt(fabs(Overlap*Overlap*Overlap))*((P1[j]-P1[tx])/RIJMag);
					
					VelRn = (PV0[tx]-PV0[j])*((P0[j]-P0[tx])/RIJMag) + (PV1[tx]-PV1[j])*((P1[j]-P1[tx])/RIJMag);
					cNormal = ContactDampingSF*sqrt(((Mass/2.0)*kNormal*sqrt(fabs(Overlap))));
					
					F0[tx] -=  cNormal*VelRn*((P0[j]-P0[tx])/RIJMag);
					F1[tx] -=  cNormal*VelRn*((P1[j]-P1[tx])/RIJMag);
					
					double theta = atan2(((P1[j]-P1[tx])/RIJMag),((P0[j]-P0[tx])/RIJMag));
					TimeContact[j*Np + tx] += 1;
					
					if(TimeContact[j*Np + tx]==1){
						
						float r = curand_uniform(&functionState);
						
						g_a[j*Np + tx] = ((r-0.5)*180*(PI/180));
						state[idvec]=functionState;
					}
					
					double passedtime = (TimeContact[j*Np + tx] > 300000) ? 1 : 0;
					
					F0[tx] -= passedtime*Mass*1000*(CSpeed*cos(theta + g_a[j*Np + tx]) - PV0[tx]);
					F1[tx] -= passedtime*Mass*1000*(CSpeed*sin(theta + g_a[j*Np + tx]) - PV1[tx]);
					
					I_C[tx] = (MaxInternalClock-1);
					double newrand = curand_uniform(&functionState);
					d_tC[tx] = newrand*100;
					state[idvec]=functionState;
					d_tCstep[tx]=2;
					
				}
				
				else{
					
					TimeContact[j*Np + tx] = 0;
					g_a[j*Np + tx] = 0;
					
					
				}
				
			}
			
			
		}
		
		__syncthreads();
		
		//---Checking whether to sense the chemo-attractant at this stage---//
		
		double notclock = (I_C[tx]!=(MaxInternalClock-1)) ? 1 : 0;
		double noCont = (All_Cont[tx]!=1) ? 1 : 0;
		
		dPhiX = 0;
		dPhiY = 0;
		dPhiMag = 0;
		
		dPhiX = PhiXp - PhiXm;
		dPhiY = PhiYp - PhiYm;
		dPhiMag = sqrt((dPhiX*dPhiX)+(dPhiY*dPhiY));
		
		double NewAngAtt = atan2(dPhiY,dPhiX);
		
		//F0[tx] += notclock*noCont*Mass*dPhiMag*alphavt*(CSpeed*cos(NewAngAtt)-PV0[tx]);
		//F1[tx] += notclock*noCont*Mass*dPhiMag*alphavt*(CSpeed*sin(NewAngAtt)-PV1[tx]);
		
		I_D[tx] = I_D[tx] - notclock*noCont*I_D[tx];
		
		//---Checking whether to tumble at this stage---//
		
		double MaxDev = (I_D[tx]==(MaxInternalDeviation-1)) ? 1 : 0;
		double MaxClock = (I_C[tx]==(MaxInternalClock-1)) ? 1 : 0;
		
		dPerX = 0;
		dPerY = 0;
		dPerMag = 0;
		NTP = Theta_P[tx] + DR_rot[tx];
		
		dPerX = cos(NTP);
		dPerY = sin(NTP);
		//dPerMag = sqrt((dPerX*dPerX)+(dPerY*dPerY));
		
		F0[tx] += MaxDev*noCont*(Mass/DTime)*((sqrt((PV0[tx]*PV0[tx])+(PV1[tx]*PV1[tx])))*dPerX - PV0[tx]);
		F1[tx] += MaxDev*noCont*(Mass/DTime)*((sqrt((PV0[tx]*PV0[tx])+(PV1[tx]*PV1[tx])))*dPerY - PV1[tx]);
		
		I_D[tx] = I_D[tx]*(1-MaxDev*MaxClock*noCont);
		
		//---If no other signals maintain a constant speed---//
		
		double notDev = (I_D[tx]!=(MaxInternalDeviation-1)) ? 1 : 0;
		
		F0[tx] += notDev*noCont*Mass*alphavt*(CSpeed*cos(Theta_P[tx])-PV0[tx]);
		F1[tx] += notDev*noCont*Mass*alphavt*(CSpeed*sin(Theta_P[tx])-PV1[tx]);
		
		//----Update the internal clocks----//
		
		d_tP[tx] += DTime;
		
		if(((d_tP[tx]>150))||(I_D[tx]==(MaxInternalDeviation-1))){
			
			double ras = curand_uniform(&functionState);
			d_tPstep[tx]=0.0;
			DR_rot[tx] = (ras-0.5)*6;
			state[idvec]=functionState;
			d_tP[tx] = 0;
			
		}
		else{
			
			d_tPstep[tx]=1.0;
			
		}
		
		
		if(d_tCstep[tx]==0){
			
			d_tCstep[tx]=2;
			d_tC[tx] = 0.0;
			
		}
		
		if(I_C[tx]==(MaxInternalClock-1)){
			
			d_tC[tx] += DTime/QBy;
			
			if(d_tC[tx]>0.1){
				
				d_tCstep[tx]=0;
				d_tC[tx] = 0;
			}
			
		}
		
		else{
			
			d_tC[tx] += DTime;
			
			if(d_tC[tx]>0.1){
				
				
				d_tCstep[tx]=0;
				d_tC[tx] = 0;
			}
			
		}
		
		
		//------CENTRAL DIFFERENCE SCHEME-------//
		
		//--Keep cells that have reached the target in their final positions--//
		
		if(P1[tx]<=LowerBoundLength){
			
			P1[tx] = LowerBoundLength-41.4e-6;
			FP1[tx] = LowerBoundLength-41.4e-6;
			
			PV0[tx] = 0;
			PV1[tx] = 0;
			
			FV0[tx] = PV0[tx];
			FV1[tx] = PV1[tx];
			
			FP0[tx] = P0[tx];
			FP1[tx] = P1[tx];
			
			PV0[tx] = FV0[tx];
			PV1[tx] = FV1[tx];
			P0[tx] = FP0[tx];
			P1[tx] = FP1[tx];
			
			ZeroVel[tx]=1;
			
		}
		
		else{
			
			FV0[tx] = PV0[tx] + ((F0[tx])/Mass)*DTime;
			FV1[tx] = PV1[tx] + ((F1[tx])/Mass)*DTime;
			
			FP0[tx] = P0[tx] + FV0[tx]*DTime;
			FP1[tx] = P1[tx] + FV1[tx]*DTime;
			
			P0[tx] = 0;
			P1[tx] = 0;
			PV0[tx] = 0;
			PV1[tx] = 0;			
			P0[tx] = FP0[tx];
			P1[tx] = FP1[tx];
			PV0[tx] = FV0[tx];
			PV1[tx] = FV1[tx];
			
		}
		
		
		//------CENTRAL DIFFERENCE SCHEME-------//
		
		//----Get simulation results for movie generation----//
		
		if((bx==9)&&(tx==1)){		
			if(CounterP==80000){
				
				
				
				for(int j=0; j<Np; j++){
					
					TimeArray[TimeDatArr*2*Np + j] = P0[j];
					
					TimeArray[TimeDatArr*2*Np + Np + j] = P1[j];
					
					VelArray[TimeDatArr*2*Np + j] = PV0[j];
					
					VelArray[TimeDatArr*2*Np + Np + j] = PV1[j];
					
				}
				
				TimeDatArr += 1;
				
				
				CounterP = 0;
				
			}
			
			CounterP += 1;
			
		}		
		//---------------------------------------------------//
		
		//----Get 80% at boundary simulation results----//
		
		if(tx==1){
			for(int h=0; h<Np; h++){
				if(P1[h]<LowerBoundLength){
					
					Eightypercent[bx] += 1;
					
				}
			}
			
			if(Eightypercent[bx]>=40){
				
				if((bx==1)){
					
					cuPrintf("Time = %f BlockIdx: %d BPhi: %f TP0: %f TP1: %f TV0: %4.10f TV1: %4.10f \n",Time,bx,phiB[bx],FP0[tx],FP1[tx],FV0[tx],FV1[tx]);
					
				}
				
				PrintOut[bx]=Time;
				
				Eightypercent[bx] = 3*Np;
			}		
			
			
			if(Eightypercent[bx]>=40){
				
				Time = TotalTime+TotalTimeStep;
				
			}
		}
		//-----------------------------------------------//
		
		//Reset vectors
		All_Cont[tx]=0;
		F0[tx] = 0;
		F1[tx] = 0;
		FV0[tx] = 0;
		FV1[tx] = 0;
		B_0[tx] = 0;
		B_1[tx] = 0;
		FP0[tx] = 0;
		FP1[tx] = 0;
		Eightypercent[bx] = 0;
		
		__syncthreads();
		
		///////////////////////////
		/////End of while loop/////
		///////////////////////////
		
	}
	
	ZV[idvec] = ZeroVel[tx];
	PPosition0[idvec]=P0[tx];
	PPosition1[idvec]=P1[tx];
	PVelocity0[idvec]=PV0[tx];
	PVelocity1[idvec]=PV1[tx];
	DTimeC[idvec]=d_tC[tx];
	InternalDeviation[idvec]=I_D[tx];
	Bgamma[idvec]=Bg_a[tx];
	BTimeContact[idvec]=BT_Contact[tx];
	gamma[idvec]=g_a[tx];
	TimeInContact[idvec]=TimeContact[tx];
	InternalClock[idvec]=I_D[tx];
	DTimeCStep[idvec]=d_tCstep[tx];
	DTimePStep[idvec]=d_tPstep[tx];
	DAngRotate[idvec]=DR_rot[tx];
	DTimeP[idvec]=d_tP[tx];

	__syncthreads();
	
	return(TimeArray);	
	
}


