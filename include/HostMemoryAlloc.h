#ifndef __HOSTMEMORYALLOC_H__
#define __HOSTMEMORYALLOC_H__

#define DEBUG_GRID 0
#define DO_TIMING 0
#include "DeviceKernel.cuh"
#include <curand_kernel.h>
#include "vector_functions.h"


class HostMemoryAlloc
{
public:
    HostMemoryAlloc(int numBlocks, int numThreads);
    ~HostMemoryAlloc();

	float* EightyPercentTime;
	double* GetTime;
	double* GetVel;
	double* GetTimeTwo;
	double* GetVelTwo;
	double* HostphiB;
	double* HostTimeArray;
	int* HostZV;
	int* HostInternalClock;
	int* HostInternalDeviation;
	int* HostEightypercent;
	float* HostPrintOut;
	float* HostPrintOutTwo;

	double* HostThetaPersist;
	double* HostDTimeCrime;
	double* HostFPosition0;
	double* HostFPosition1;
	double* HostPPosition0;
	double* HostPPosition1;
	double* HostFVelocity0;
	double* HostFVelocity1;
	double* HostPVelocity0;
	double* HostPVelocity1;

	double* HostFMig0;
	double* HostFMig1;
	double* HostBIJ0;
	double* HostBIJ1;
	float* HostDTimePrime;
	
	double* HostDAngRotate;
	float* HostDTimePrimeStep;
	float* HostDTimeCrimeStep;
	float* HostBgamma;
	long int* HostTimeInContact;
	long int* HostBTimeContact;
	long int* HostBTimeInContact;
	float* Hostgamma;
	int* HostAllContacts;

	void Initialise(int Num);
    void XData(int numThreads, int numBlocks, int Num);
    int getNumParticles() const { return m_numParticles; }
    void randkernel(int numThreads, int numBlocks);
    int m_numParticles;
	void copyAfromD(int Num);

protected: // methods


	float* DTimeCrime;
	double* FPosition0;
	double* FPosition1;
	double* PPosition0;
	double* PPosition1;
	double* FVelocity0;
	double* FVelocity1;
	double* PVelocity0;
	double* PVelocity1;

	double* FMig0;
	double* FMig1;
	double* BIJ0;
	double* BIJ1;
	float* DTimePrime;

	float* Timer;
	double* DAngRotate;
	float* DTimePrimeStep;
	float* DTimeCrimeStep;
	double* Bgamma;
	long int* TimeInContact;
	long int* BTimeContact;
	long int* BTimeInContact;
	float* gamma;
	int* AllContacts;
	int* ZV;	
	double* TimeArray;
	double* VelArray;
	double* phiB;
	int* InternalClock;
	int* InternalDeviation;
	int* Eightypercent;
	float* PrintOut;
	
	double* ThetaPersist;
       
	curandState *devStates;
	
	SimParams m_params;

};

#endif // __HostMemoryAlloc_H__
