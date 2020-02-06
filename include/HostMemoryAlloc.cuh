#include <DeviceKernel.cuh>
#include <curand_kernel.h>
extern "C"
{

void cudaInit(int argc, char **argv);

void allocateArray(void **devPtr, size_t size);
void freeArray(void *devPtr);

void threadSync();

void copyArrayFromDevice(void* host, const void* device, int size);

void copyArrayToDevice(void* device, const void* host, int offset, int size);

void setParameters(SimParams *hostParams);

void HostCallGo(uint numThreads, uint numBlocks);

void HostCallPrintX(uint numThreads, uint numBlocks, curandState *devStates, int* InternalClock, double* ThetaPersist,  int* 
InternalDeviation, float* DTimeCrime,  double* FPosition0,  double* FPosition1,  double* PPosition0,  double* PPosition1, double* 
FVelocity0,  double* FVelocity1,  double* PVelocity0,  double* PVelocity1,double* FMig0,double* FMig1,double* BIJ0,double* BIJ1,float* 
DTimePrime,double* DAngRotate,float* DTimePrimeStep,float* DTimeCrimeStep,double* Bgamma,long int* 
TimeInContact,long int* BTimeContact,long int* BTimeInContact,float* gamma,int* AllContacts, double* TimeArray, double* phiB,int* 
Eightypercent, float* PrintOut, int* ZV, double* VelArray);
	
void HostCallPrintX1(uint numThreads, uint numBlocks, curandState* devStates, int* InternalClock,double* ThetaPersist,int* InternalDeviation,float* 
						 DTimeCrime,double*  FPosition0, double* FPosition1,  double* PPosition0,double*  PPosition1,double* FVelocity0, double* FVelocity1, double* PVelocity0, 
						 double* PVelocity1,double* FMig0,double* FMig1,double* BIJ0,double* BIJ1,float* DTimePrime,double* DAngRotate,float* DTimePrimeStep,float* 
						 DTimeCrimeStep,double* Bgamma,long int* TimeInContact,long int* BTimeContact,long int* BTimeInContact,float* 
gamma,int* 
AllContacts,double* 
						 TimeArray, double* 
					 Bphi,int* Eightypercent, float* PrintOut, int* ZV, double* VelArray);

void allocateMatrix(void **pointPtr, size_t pitch, size_t size, int height);

void setkernel(curandState *devStates, int numThreads, int numBlocks);


};
