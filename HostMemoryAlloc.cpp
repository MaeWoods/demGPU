/////////////////////////////////////////////////////////////////////////////
// Name:         HostMemoryAlloc.cpp
// Purpose:      Simulating the discrete element model on a GPU
// Author:       Mae Woods UCL
// Modified by:
// Created:      Jan/2014
// Copyright:    Open source
// Licence:      None
// Dependencies: CUDA see http://www.nvidia.co.uk/object/cuda-parallel-computing-uk.html
/////////////////////////////////////////////////////////////////////////////

#include "HostMemoryAlloc.h"
#include "HostMemoryAlloc.cuh"
#include "DeviceKernel.cuh"
#include <iostream>
#include <fstream>
#include <iostream>
#include <assert.h>
#include <math.h>
#include <curand_kernel.h>
#include <memory.h>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <cstdlib>
#include <cstdio>
#ifndef CUDART_PI_F
#define CUDART_PI_F         3.141592654f
#endif
using namespace std;

HostMemoryAlloc::HostMemoryAlloc(int numBlocks, int numThreads)
{
    m_params.numBlocks=numBlocks;
    m_params.numThreads=numThreads;
	int Np = 50;
	
}
 

HostMemoryAlloc::~HostMemoryAlloc()
{

}

void HostMemoryAlloc::randkernel(int numThreads, int numBlocks)
{

	cudaMalloc((void **)&devStates, numBlocks*50*sizeof(curandState));
	setkernel(devStates, numThreads, numBlocks);

}

		////////////////////////////////////////////////////////////
		// Allocate Vectors									      //	
		// The function Initialise initialises the vectors on the //
		// host (CPU) and creates memory on the device (GPU)      //
		// through the function allocateArray                     //
		////////////////////////////////////////////////////////////

void HostMemoryAlloc::Initialise(int runNo){
	
	if(runNo==1){

	int Np = 50;
	int BlockParam = 10;
	int memdouble = sizeof(double) * Np * BlockParam;
	int memfloat = sizeof(float) * Np * BlockParam;
	int memOutput = sizeof(double) * BlockParam;
	int memOutputI = sizeof(int) * BlockParam;
	int memOutputF = sizeof(float) * BlockParam;
	int memint = sizeof(int) * Np * BlockParam;
	int memMatdouble = sizeof(float) * (Np * Np * BlockParam);
	int memMatint = sizeof(int) * (Np * Np * BlockParam);
	int memMatfloat = sizeof(float) * (Np * Np * BlockParam);
	int memOutputEighty = sizeof(int) * Np *  BlockParam;
	int memlongint = sizeof(long int) * Np * BlockParam;
	int memMatlongint = sizeof(long int) * (Np * Np * BlockParam);
	int memTA = sizeof(double) * 2*Np*395;

	allocateArray((void**)&TimeArray, memTA);
	allocateArray((void**)&VelArray, memTA);
	allocateArray((void**)&phiB, memOutput);
	allocateArray((void**)&Eightypercent, memOutputEighty);
	allocateArray((void**)&PrintOut, memOutputF);
	allocateArray((void**)&ZV, memint);	
	allocateArray((void**)&InternalClock, memint);
	allocateArray((void**)&ThetaPersist, memdouble);
	allocateArray((void**)&InternalDeviation, memint);
	allocateArray((void**)&CtTime, memdouble);
	allocateArray((void**)&FPosition0, memdouble);
	allocateArray((void**)&FPosition1, memdouble);
	allocateArray((void**)&PPosition0, memdouble);
	allocateArray((void**)&PPosition1, memdouble);
	allocateArray((void**)&FVelocity0, memdouble);
	allocateArray((void**)&FVelocity1, memdouble);
	allocateArray((void**)&PVelocity0, memdouble);
	allocateArray((void**)&PVelocity1, memdouble);
	allocateArray((void**)&AllContacts, memint);
	allocateArray((void**)&FMig0, memdouble);
	allocateArray((void**)&FMig1, memdouble);
	allocateArray((void**)&BIJ0, memdouble);
	allocateArray((void**)&BIJ1, memdouble);
	allocateArray((void**)&DtTime, memfloat);
	allocateArray((void**)&DAngRotate, memdouble);
	allocateArray((void**)&DtTimeStep, memfloat);
	allocateArray((void**)&CtTimeStep, memfloat);
	allocateArray((void**)&Bgamma, memdouble);
	allocateArray((void**)&BTimeContact, memlongint);
	allocateArray((void**)&BTimeInContact, memlongint);
	allocateArray((void**)&TimeInContact, memMatlongint);
	allocateArray((void**)&gamma, memMatfloat);

	
	HostphiB = new double[memOutput];
	HostTimeInContact= new long int[memMatlongint];
	Hostgamma= new float[memMatfloat];

	}
	
	else if(runNo==2){
	
		//If runNo=2, input files determine the state of the system.
		//Here, these input files are used to initialise the vectors
		//on the host. The values of the vectors are then copied over
		//to memory on the GPU
		
		int Np = 50;
		int BlockParam = 10;
		int numThreads = 50;
		int numBlocks = 10;
		int memlongint = sizeof(long int) * Np * BlockParam;	
		int memMatlongint = sizeof(long int) * (Np * Np * BlockParam);		
		int memF = sizeof(float) * Np * BlockParam;
		int memint = sizeof(int) * Np * BlockParam;
		int memdouble = sizeof(double) * Np * BlockParam;
		int memfloat = sizeof(float) * Np * BlockParam;
		int memOutput = sizeof(double) * BlockParam;
		int memOutputI = sizeof(int) * BlockParam;
		int memOutputF = sizeof(float) * BlockParam;
		int memMatdouble = sizeof(float) * (Np * Np * BlockParam);
		int memMatint = sizeof(int) * (Np * Np * BlockParam);
		int memMatfloat = sizeof(float) * (Np * Np * BlockParam);
		int memOutputEighty = sizeof(int) * Np *  BlockParam;
		int memSizeOut = sizeof(double) * 2*Np*395;
		
		EightyPercentTime = new float[BlockParam];
		memset(EightyPercentTime, 0, sizeof(float) * BlockParam);
		
		HostPrintOutTwo = new float[BlockParam];
		memset(HostPrintOutTwo, 0, memOutputF);
		
		HostZV = new int[Np * BlockParam];
		memset(HostZV, 0, memint);
		
		GetTime = new double[2*50*395];
		memset(GetTime, 0, memSizeOut);
		
		GetVel = new double[2*50*395];
		memset(GetVel, 0, memSizeOut);
		
		HostPPosition0 = new double[Np * BlockParam];
		memset(HostPPosition0, 0, memdouble);
		
		HostPPosition1 = new double[Np * BlockParam];
		memset(HostPPosition1, 0, memdouble);
		
		HostPVelocity0 = new double[Np * BlockParam];
		memset(HostPVelocity0, 0, memdouble);
		
		HostPVelocity1 = new double[Np * BlockParam];
		memset(HostPVelocity1, 0, memdouble);
		
		
		HostInternalDeviation = new int[Np * BlockParam];
		memset(HostInternalDeviation,0,memint);
		
		HostCtTime = new double[Np * BlockParam];
		memset(HostCtTime,0,memdouble);
		
		HostBgamma = new float[Np * BlockParam];
		memset(HostBgamma,0,memfloat);
		
		HostBTimeContact = new long int[Np * BlockParam];
		memset(HostBTimeContact,0,memlongint);
		
		Hostgamma = new float[Np * Np * BlockParam];
		memset(Hostgamma,0,memfloat);
		
		HostTimeInContact = new long int[Np * Np * BlockParam];
		memset(HostTimeInContact,0,memMatlongint);
		
		HostInternalClock = new int[Np * BlockParam];
		memset(HostInternalClock,0,memint);
		
		HostCtTimeStep = new float[Np * BlockParam];
		memset(HostCtTimeStep,0,memfloat);
		
		HostDtTimeStep = new float[Np * BlockParam];
		memset(HostDtTimeStep,0,memfloat);
		
		HostDAngRotate = new double[Np * BlockParam];
		memset(HostDAngRotate,0,memdouble);
		
		HostDtTime = new float[Np * BlockParam];
		memset(HostDtTime,0,memfloat);
		
		
		
		ifstream myEightyPercentTime;              
		
		char place0[100];
		
		int n0;
		
		n0 = sprintf(place0,"EightyPercentTime%d",runNo);  
		
		myEightyPercentTime.open(place0);
		
		string line0;
		
		if (myEightyPercentTime.is_open())
		{
			int counter = 0;
			while ( myEightyPercentTime.good() )
			{
				
				// Getline, delimited by ' '
				// Note that string use " and that char use ' 
				
				getline (myEightyPercentTime,line0,' ');
				
				// Convert String to Char 
				// Same method as above 
				
				char* LineChar =new char[line0.size()+1];
				
				LineChar[line0.size()]=0;
				
				memcpy(LineChar,line0.c_str(),line0.size());                        
				
				// Convert to double
				
				double value = strtod(LineChar,NULL);
				
				if(counter<BlockParam){
					EightyPercentTime[counter] = (float)(value);
				}
				
				counter += 1;
				
			}
			
			myEightyPercentTime.close();
			
		}
				
		
		ifstream myPrintOut;              
		
		char place1[100];
		
		int n1;
		
		n1 = sprintf(place1,"PrintOut%d",runNo);  
		
		myPrintOut.open(place1);
		
		string line;
		
		if (myPrintOut.is_open())
		{
			int counter = 0;
			while ( myPrintOut.good() )
			{
				
				// Getline, delimited by ' '
				// Note that string use " and that char use ' 
				
				getline (myPrintOut,line,' ');
				
				// Convert String to Char 
				// Same method as above 
				
				char* LineChar =new char[line.size()+1];
				
				LineChar[line.size()]=0;
				
				memcpy(LineChar,line.c_str(),line.size());                        
				
				// Convert to double
				
				double value = strtod(LineChar,NULL);
				
				if(counter<BlockParam){
					HostPrintOutTwo[counter] = (float)(value);
				}
				
				counter += 1;
				
			}
			
			myPrintOut.close();
			
		}
		
		
		ifstream myZV;             
		
		char place2[100];
		
		int n2;
		
		n2 = sprintf(place2,"ZV%d",runNo);   
		
		myZV.open(place2);
		
		string line1;
		
		if (myZV.is_open())
		{
			int counter = 0;
			while ( myZV.good() )
			{
				
				// Getline, delimited by ' '
				// Note that string use " and that char use ' 
				
				getline (myZV,line1,' ');
				
				// Convert String to Char 
				// Same method as above 
				
				char* LineChar =new char[line1.size()+1];
				
				LineChar[line1.size()]=0;
				
				memcpy(LineChar,line1.c_str(),line1.size());                        
				
				// Convert to double
				
				double value = strtod(LineChar,NULL);
				
				if(counter<Np * BlockParam){
					HostZV[counter] = (int)value;
				}
				
				counter += 1;
				
			}
			
			myZV.close();
			
		}
		
		
		ifstream myTime;     
		
		char place3[100];
		
		int n3;
		
		n3 = sprintf(place3,"Time%d",runNo);
		
		myTime.open(place3);
		
		string line2;
		
		if (myTime.is_open())
		{
			int counter = 0;
			while ( myTime.good() )
			{
				
				// Getline, delimited by ' '
				// Note that string use " and that char use ' 
				
				getline (myTime,line2,' ');
				
				// Convert String to Char 
				// Same method as above 
				
				char* LineChar =new char[line2.size()+1];
				
				LineChar[line2.size()]=0;
				
				memcpy(LineChar,line2.c_str(),line2.size());                        
				
				// Convert to double
				
				double value = strtod(LineChar,NULL);
				
				if(counter<2*50*395){
					GetTime[counter] = value;
				}
				
				counter += 1;
				
			}
			
			myTime.close();
			
		}
		
		
		ifstream myVel;         
		
		char place4[100];
		
		int n4;
		
		n4 = sprintf(place4,"Vel%d",runNo);
		
		myVel.open(place4);
		
		string line3;
		
		if (myVel.is_open())
		{
			int counter = 0;
			while ( myVel.good() )
			{
				
				// Getline, delimited by ' '
				// Note that string use " and that char use ' 
				
				getline (myVel,line3,' ');
				
				// Convert String to Char 
				// Same method as above 
				
				char* LineChar =new char[line3.size()+1];
				
				LineChar[line3.size()]=0;
				
				memcpy(LineChar,line3.c_str(),line3.size());                        
				
				// Convert to double
				
				double value = strtod(LineChar,NULL);
				
				if(counter<2*50*395){
					GetVel[counter] = value;
				}
				
				counter += 1;
				
			}
			
			myVel.close();
			
		}
		
		
		ifstream myPosition0;    
		
		char place5[100];
		
		int n5;
		
		n5 = sprintf(place5,"Position0%d",runNo);
		
		myPosition0.open(place5);
		
		string line4;
		
		if (myPosition0.is_open())
		{
			int counter = 0;
			while ( myPosition0.good() )
			{
				
				// Getline, delimited by ' '
				// Note that string use " and that char use ' 
				
				getline (myPosition0,line4,' ');
				
				// Convert String to Char 
				// Same method as above 
				
				char* LineChar =new char[line4.size()+1];
				
				LineChar[line4.size()]=0;
				
				memcpy(LineChar,line4.c_str(),line4.size());                        
				
				// Convert to double
				
				double value = strtod(LineChar,NULL);
				
				if(counter<Np * BlockParam){
					HostPPosition0[counter] = value;
				}
				
				counter += 1;
				
			}
			
			myPosition0.close();
			
		}
		
		
		
		ifstream myPosition1;   
		
		char place6[100];
		
		int n6;
		
		n6 = sprintf(place6,"Position1%d",runNo);
		
		myPosition1.open(place6);
		
		string line5;
		
		if (myPosition1.is_open())
		{
			int counter = 0;
			while ( myPosition1.good() )
			{
				
				// Getline, delimited by ' '
				// Note that string use " and that char use ' 
				
				getline (myPosition1,line5,' ');
				
				// Convert String to Char 
				// Same method as above 
				
				char* LineChar =new char[line5.size()+1];
				
				LineChar[line5.size()]=0;
				
				memcpy(LineChar,line5.c_str(),line5.size());                        
				
				// Convert to double
				
				double value = strtod(LineChar,NULL);
				
				if(counter<Np * BlockParam){
					HostPPosition1[counter] = value;
				}
				
				counter += 1;
				
			}
			
			myPosition1.close();
			
		}
		
		
		ifstream myPVelocity0;
		
		char place7[100];
		
		int n7;
		
		n7 = sprintf(place7,"PVelocity0%d",runNo);
		
		myPVelocity0.open(place7);
		
		string line6;
		
		if (myPVelocity0.is_open())
		{
			int counter = 0;
			while ( myPVelocity0.good() )
			{
				
				// Getline, delimited by ' '
				// Note that string use " and that char use ' 
				
				getline (myPVelocity0,line6,' ');
				
				// Convert String to Char 
				// Same method as above 
				
				char* LineChar =new char[line6.size()+1];
				
				LineChar[line6.size()]=0;
				
				memcpy(LineChar,line6.c_str(),line6.size());                        
				
				// Convert to double
				
				double value = strtod(LineChar,NULL);
				
				if(counter<Np * BlockParam){
					HostPVelocity0[counter] = value;
				}
				
				counter += 1;
				
			}
			
			myPVelocity0.close();
			
		}
		
				
		ifstream myPVelocity1;  
		
		char place8[100];
		
		int n8;
		
		n8 = sprintf(place8,"PVelocity1%d",runNo);
		
		myPVelocity1.open(place8);
		
		string line7;
		
		if (myPVelocity1.is_open())
		{
			int counter = 0;
			while ( myPVelocity1.good() )
			{
				
				// Getline, delimited by ' '
				// Note that string use " and that char use ' 
				
				getline (myPVelocity1,line7,' ');
				
				// Convert String to Char 
				// Same method as above 
				
				char* LineChar =new char[line7.size()+1];
				
				LineChar[line7.size()]=0;
				
				memcpy(LineChar,line7.c_str(),line7.size());                        
				
				// Convert to double
				
				double value = strtod(LineChar,NULL);
				
				if(counter<Np * BlockParam){
					HostPVelocity1[counter] = value;
				}
				
				counter += 1;
				
			}
			
			myPVelocity1.close();
			
		}
		
		
		ifstream myDtTime;  
		
		char place9[100];
		
		int n9;
		
		n9 = sprintf(place9,"DtTime%d",runNo);
		
		myDtTime.open(place9);
		
		string line9;
		
		if (myDtTime.is_open())
		{
			int counter = 0;
			while ( myDtTime.good() )
			{
				
				// Getline, delimited by ' '
				// Note that string use " and that char use ' 
				
				getline (myDtTime,line9,' ');
				
				// Convert String to Char 
				// Same method as above 
				
				char* LineChar =new char[line9.size()+1];
				
				LineChar[line9.size()]=0;
				
				memcpy(LineChar,line9.c_str(),line9.size());                        
				
				// Convert to double
				
				double value = strtod(LineChar,NULL);
				
				if(counter<Np * BlockParam){
					HostDtTime[counter] = (float)value;
				}
				
				counter += 1;
				
			}
			
			myDtTime.close();
			
		
		}
		
		
				
		ifstream myDAngRotate;  
		
		char place10[100];
		
		int n10;
		
		n10 = sprintf(place10,"DAngRotate%d",runNo);
		
		myDAngRotate.open(place10);
		
		string line10;
		
		if (myDAngRotate.is_open())
		{
			int counter = 0;
			while ( myDAngRotate.good() )
			{
				
				// Getline, delimited by ' '
				// Note that string use " and that char use ' 
				
				getline (myDAngRotate,line10,' ');
				
				// Convert String to Char 
				// Same method as above 
				
				char* LineChar =new char[line10.size()+1];
				
				LineChar[line10.size()]=0;
				
				memcpy(LineChar,line10.c_str(),line10.size());                        
				
				// Convert to double
				
				double value = strtod(LineChar,NULL);
				
				if(counter<Np * BlockParam){
					HostDAngRotate[counter] = value;
				}
				
				counter += 1;
				
			}
			
			myDAngRotate.close();
			
		}
		
		
		ifstream myDtTimeStep;  
		
		char place11[100];
		
		int n11;
		
		n11 = sprintf(place11,"DtTimeStep%d",runNo);
		
		myDtTimeStep.open(place11);
		
		string line11;
		
		if (myDtTimeStep.is_open())
		{
			int counter = 0;
			while ( myDtTimeStep.good() )
			{
				
				// Getline, delimited by ' '
				// Note that string use " and that char use ' 
				
				getline (myDtTimeStep,line11,' ');
				
				// Convert String to Char 
				// Same method as above 
				
				char* LineChar =new char[line11.size()+1];
				
				LineChar[line11.size()]=0;
				
				memcpy(LineChar,line11.c_str(),line11.size());                        
				
				// Convert to double
				
				double value = strtod(LineChar,NULL);
				
				if(counter<Np * BlockParam){
					HostDtTimeStep[counter] = (float)value;
				}
				
				counter += 1;
				
			}
			
			myDtTimeStep.close();
			
		}
		
		
		ifstream myCtTimeStep;  
		
		char place12[100];
		
		int n12;
		
		n12 = sprintf(place12,"CtTimeStep%d",runNo);
		
		myCtTimeStep.open(place12);
		
		string line12;
		
		if (myCtTimeStep.is_open())
		{
			int counter = 0;
			while ( myCtTimeStep.good() )
			{
				
				// Getline, delimited by ' '
				// Note that string use " and that char use ' 
				
				getline (myCtTimeStep,line12,' ');
				
				// Convert String to Char 
				// Same method as above 
				
				char* LineChar =new char[line12.size()+1];
				
				LineChar[line12.size()]=0;
				
				memcpy(LineChar,line12.c_str(),line12.size());                        
				
				// Convert to double
				
				double value = strtod(LineChar,NULL);
				
				if(counter<Np * BlockParam){
					HostCtTimeStep[counter] = (float)value;
				}
				
				counter += 1;
				
			}
			
			myCtTimeStep.close();
			
		}
		
		
		ifstream myInternalClock;  
		
		char place13[100];
		
		int n13;
		
		n13 = sprintf(place13,"InternalClock%d",runNo);
		
		myInternalClock.open(place13);
		
		string line13;
		
		if (myInternalClock.is_open())
		{
			int counter = 0;
			while ( myInternalClock.good() )
			{
				
				// Getline, delimited by ' '
				// Note that string use " and that char use ' 
				
				getline (myInternalClock,line13,' ');
				
				// Convert String to Char 
				// Same method as above 
				
				char* LineChar =new char[line13.size()+1];
				
				LineChar[line13.size()]=0;
				
				memcpy(LineChar,line13.c_str(),line13.size());                        
				
				// Convert to double
				
				double value = strtod(LineChar,NULL);
				
				if(counter<Np * BlockParam){
					HostInternalClock[counter] = (int)value;
				}
				
				counter += 1;
				
			}
			
			myInternalClock.close();
			
		}
		
		
		
		ifstream myTimeInContact;  
		
		char place14[100];
		
		int n14;
		
		n14 = sprintf(place14,"TimeInContact%d",runNo);
		
		myTimeInContact.open(place14);
		
		string line14;
		
		if (myTimeInContact.is_open())
		{
			int counter = 0;
			while ( myTimeInContact.good() )
			{
				
				// Getline, delimited by ' '
				// Note that string use " and that char use ' 
				
				getline (myTimeInContact,line14,' ');
				
				// Convert String to Char 
				// Same method as above 
				
				char* LineChar =new char[line14.size()+1];
				
				LineChar[line14.size()]=0;
				
				memcpy(LineChar,line14.c_str(),line14.size());                        
				
				// Convert to double
				
				double value = strtod(LineChar,NULL);
				
				if(counter<Np * Np * BlockParam){
					HostTimeInContact[counter] = (int)value;
				}
				
				counter += 1;
				
			}
			
			myTimeInContact.close();
			
		}
		
		
		
		ifstream mygamma;  
		
		char place15[100];
		
		int n15;
		
		n15 = sprintf(place15,"gamma%d",runNo);
		
		mygamma.open(place15);
		
		string line15;
		
		if (mygamma.is_open())
		{
			int counter = 0;
			while ( mygamma.good() )
			{
				
				// Getline, delimited by ' '
				// Note that string use " and that char use ' 
				
				getline (mygamma,line15,' ');
				
				// Convert String to Char 
				// Same method as above 
				
				char* LineChar =new char[line15.size()+1];
				
				LineChar[line15.size()]=0;
				
				memcpy(LineChar,line15.c_str(),line15.size());                        
				
				// Convert to double
				
				double value = strtod(LineChar,NULL);
				
				if(counter<Np * Np * BlockParam){
					Hostgamma[counter] = (float)value;
				}
				
				counter += 1;
				
			}
			
			mygamma.close();
			
		}
		
		
		
		ifstream myBTimeContact;  
		
		char place25[100];
		
		int n25;
		
		n25 = sprintf(place25,"BTimeContact%d",runNo);
		
		myBTimeContact.open(place25);
		
		string line25;
		
		if (myBTimeContact.is_open())
		{
			int counter = 0;
			while ( myBTimeContact.good() )
			{
				
				// Getline, delimited by ' '
				// Note that string use " and that char use ' 
				
				getline (myBTimeContact,line25,' ');
				
				// Convert String to Char 
				// Same method as above 
				
				char* LineChar =new char[line25.size()+1];
				
				LineChar[line25.size()]=0;
				
				memcpy(LineChar,line25.c_str(),line25.size());                        
				
				// Convert to double
				
				double value = strtod(LineChar,NULL);
				
				if(counter<Np * BlockParam){
					HostBTimeContact[counter] = (long int)value;
				}
				
				counter += 1;
				
			}
			
			myBTimeContact.close();
			
		}
		
		
		
		
		ifstream myBgamma;  
		
		char place16[100];
		
		int n16;
		
		n16 = sprintf(place16,"Bgamma%d",runNo);
		
		myBgamma.open(place16);
		
		string line16;
		
		if (myBgamma.is_open())
		{
			int counter = 0;
			while ( myBgamma.good() )
			{
				
				// Getline, delimited by ' '
				// Note that string use " and that char use ' 
				
				getline (myBgamma,line16,' ');
				
				// Convert String to Char 
				// Same method as above 
				
				char* LineChar =new char[line16.size()+1];
				
				LineChar[line16.size()]=0;
				
				memcpy(LineChar,line16.c_str(),line16.size());                        
				
				// Convert to double
				
				double value = strtod(LineChar,NULL);
				
				if(counter<Np * BlockParam){
					HostBgamma[counter] = (float)value;
				}
				
				counter += 1;
				
			}
			
			myBgamma.close();
			
		}
		
		
		
		ifstream myCtTime;  
		
		char place17[100];
		
		int n17;
		
		n17 = sprintf(place17,"CtTime%d",runNo);
		
		myCtTime.open(place17);
		
		string line17;
		
		if (myCtTime.is_open())
		{
			int counter = 0;
			while ( myCtTime.good() )
			{
				
				// Getline, delimited by ' '
				// Note that string use " and that char use ' 
				
				getline (myCtTime,line17,' ');
				
				// Convert String to Char 
				// Same method as above 
				
				char* LineChar =new char[line17.size()+1];
				
				LineChar[line17.size()]=0;
				
				memcpy(LineChar,line17.c_str(),line17.size());                        
				
				// Convert to double
				
				double value = strtod(LineChar,NULL);
				
				if(counter<Np * BlockParam){
					HostCtTime[counter] = value;
				}
				
				counter += 1;
				
			}
			
			myCtTime.close();
			
		}
		
		
		
		ifstream myInternalDeviation;  
		
		char place18[100];
		
		int n18;
		
		n18 = sprintf(place18,"InternalDeviation%d",runNo);
		
		myInternalDeviation.open(place18);
		
		string line18;
		
		if (myInternalDeviation.is_open())
		{
			int counter = 0;
			while ( myInternalDeviation.good() )
			{
				
				// Getline, delimited by ' '
				// Note that string use " and that char use ' 
				
				getline (myInternalDeviation,line18,' ');
				
				// Convert String to Char 
				// Same method as above 
				
				char* LineChar =new char[line18.size()+1];
				
				LineChar[line18.size()]=0;
				
				memcpy(LineChar,line18.c_str(),line18.size());                        
				
				// Convert to double
				
				double value = strtod(LineChar,NULL);
				
				if(counter<Np * BlockParam){
					HostInternalDeviation[counter] = (int)value;
				}
				
				counter += 1;
				
			}
			
			myInternalDeviation.close();
			
		}
		

		
		int memTA = sizeof(double) * 2*Np*395;
		allocateArray((void**)&TimeArray, memTA);
		allocateArray((void**)&VelArray, memTA);
		allocateArray((void**)&phiB, memOutput);
		allocateArray((void**)&Eightypercent, memOutputEighty);
		allocateArray((void**)&PrintOut, memOutputF);
		allocateArray((void**)&ZV, memint);	
		allocateArray((void**)&InternalClock, memint);
		allocateArray((void**)&ThetaPersist, memdouble);
		allocateArray((void**)&InternalDeviation, memint);
		allocateArray((void**)&CtTime, memdouble);		
		allocateArray((void**)&FPosition0, memdouble);
		allocateArray((void**)&FPosition1, memdouble);
		allocateArray((void**)&PPosition0, memdouble);
		allocateArray((void**)&PPosition1, memdouble);
		allocateArray((void**)&FVelocity0, memdouble);
		allocateArray((void**)&FVelocity1, memdouble);
		allocateArray((void**)&PVelocity0, memdouble);
		allocateArray((void**)&PVelocity1, memdouble);
		allocateArray((void**)&AllContacts, memint);
		allocateArray((void**)&FMig0, memdouble);
		allocateArray((void**)&FMig1, memdouble);		
		allocateArray((void**)&BIJ0, memdouble);
		allocateArray((void**)&BIJ1, memdouble);		
		allocateArray((void**)&DtTime, memfloat);
		allocateArray((void**)&DAngRotate, memdouble);
		allocateArray((void**)&DtTimeStep, memfloat);
		allocateArray((void**)&CtTimeStep, memfloat);
		allocateArray((void**)&Bgamma, memdouble);
		allocateArray((void**)&BTimeContact, memlongint);
		allocateArray((void**)&BTimeInContact, memlongint);	
		allocateArray((void**)&TimeInContact, memMatlongint);
		allocateArray((void**)&gamma, memMatfloat);
		
		
		copyArrayToDevice(PPosition0, HostPPosition0, 0, memdouble);
		copyArrayToDevice(PPosition1, HostPPosition1, 0, memdouble);
		copyArrayToDevice(PVelocity0, HostPVelocity0, 0, memdouble);
		copyArrayToDevice(PVelocity1, HostPVelocity1, 0, memdouble);
		copyArrayToDevice(PrintOut, HostPrintOutTwo, 0, memOutputF);
		copyArrayToDevice(ZV, HostZV, 0, memint);
		copyArrayToDevice(TimeArray, GetTime, 0, memSizeOut);
		copyArrayToDevice(VelArray, GetVel, 0, memSizeOut);
		
		
		copyArrayToDevice(InternalDeviation,HostInternalDeviation,0,memint);
		copyArrayToDevice(CtTime,HostCtTime,0,memdouble);
		copyArrayToDevice(Bgamma,HostBgamma,0,memfloat);
		copyArrayToDevice(BTimeContact,HostBTimeContact,0,memlongint);
		copyArrayToDevice(gamma,Hostgamma,0,memfloat);
		copyArrayToDevice(TimeInContact,HostTimeInContact,0,memMatint);
		copyArrayToDevice(InternalClock,HostInternalClock,0,memint);
		copyArrayToDevice(CtTimeStep,HostCtTimeStep,0,memfloat);
		copyArrayToDevice(DtTimeStep,HostDtTimeStep,0,memfloat);
		copyArrayToDevice(DAngRotate,HostDAngRotate,0,memdouble);
		copyArrayToDevice(DtTime,HostDtTime,0,memfloat);
		
		

	}
		///////////////////////////////
		//////Allocate Vectors/////////
		///////////////////////////////
	
}

void HostMemoryAlloc::XData(int numThreads, int numBlocks, int runNo)
{

	int BlockParam = 10;

	if(runNo==1){
		
		HostCallPrintX(numThreads, numBlocks, devStates, InternalClock, ThetaPersist,  InternalDeviation, CtTime,  FPosition0,  FPosition1,  PPosition0,  
PPosition1, FVelocity0,  FVelocity1,  PVelocity0,  
PVelocity1,FMig0,FMig1,BIJ0,BIJ1,DtTime,DAngRotate,DtTimeStep,CtTimeStep,Bgamma,TimeInContact,BTimeContact,BTimeInContact,gamma,AllContacts,TimeArray,phiB,Eightypercent,PrintOut,ZV,VelArray);

	}
	
	else if(runNo==2){
		
		
		
		HostCallPrintX1(numThreads, numBlocks, devStates, InternalClock, ThetaPersist,  InternalDeviation, CtTime,  FPosition0,  FPosition1,  PPosition0,  PPosition1, FVelocity0,  FVelocity1,  PVelocity0, PVelocity1,FMig0,FMig1,BIJ0,BIJ1,DtTime,DAngRotate,DtTimeStep,CtTimeStep,Bgamma,TimeInContact,BTimeContact,BTimeInContact,gamma,AllContacts,TimeArray,phiB,Eightypercent,PrintOut,ZV,VelArray);
		
	
		
	}
		
}

		////////////////////////////////////////////////////////////
		// Return Data								              //	
		// The function copies data from the device (GPU) to      //
		// the host (CPU) and exports the data to the chosen      //
		// file system                                            //
		////////////////////////////////////////////////////////////


void HostMemoryAlloc::copyAfromD(int runNo){
	
	int Np = 50;
	int BlockParam = 10;
	
	if(runNo==1){
		
		
		int numThreads = 50;
		int numBlocks = 10;
		
		int memdouble = sizeof(double) * Np * BlockParam;
        int memfloat = sizeof(float) * Np * BlockParam;
        int memOutput = sizeof(double) * BlockParam;
        int memOutputI = sizeof(int) * BlockParam;
        int memOutputF = sizeof(float) * BlockParam;
        int memMatdouble = sizeof(float) * (Np * Np * BlockParam);
        int memMatint = sizeof(int) * (Np * Np * BlockParam);
        int memMatfloat = sizeof(float) * (Np * Np * BlockParam);
        int memOutputEighty = sizeof(int) * Np *  BlockParam;
		int memMatlongint = sizeof(long int) * (Np * Np * BlockParam);


		int memF = sizeof(float) * Np * BlockParam;
		int memint = sizeof(int) * Np * BlockParam;
		int memlongint = sizeof(long int) * Np * BlockParam;
		EightyPercentTime = new float[BlockParam];
		memset(EightyPercentTime, 0, sizeof(float) * BlockParam);
		
		HostPrintOut = new float[BlockParam];
		memset(HostPrintOut, 0, memOutputF);
		
		copyArrayFromDevice(HostPrintOut,PrintOut,memOutputF);
		
		HostZV = new int[Np * BlockParam];
        	memset(HostZV, 0, memint);
		
		copyArrayFromDevice(HostZV,ZV,memint);
		
		int memSizeOut = sizeof(double) * 2*Np*395;
		
		GetTime = new double[2*50*395];
		memset(GetTime, 0, memSizeOut);
		
		copyArrayFromDevice(GetTime, TimeArray, memSizeOut);
		
		GetVel = new double[2*50*395];
		memset(GetVel, 0, memSizeOut);
		
		copyArrayFromDevice(GetVel, VelArray, memSizeOut);
		
		HostPPosition0 = new double[Np * BlockParam];
        memset(HostPPosition0, 0, memdouble);
        copyArrayFromDevice(HostPPosition0, PPosition0, memdouble);
		
        HostPPosition1 = new double[Np * BlockParam];
        memset(HostPPosition1, 0, memdouble);
        copyArrayFromDevice(HostPPosition1, PPosition1, memdouble);
		
        HostPVelocity0 = new double[Np * BlockParam];
        memset(HostPVelocity0, 0, memdouble);
        copyArrayFromDevice(HostPVelocity0, PVelocity0, memdouble);
		
        HostPVelocity1 = new double[Np * BlockParam];
        memset(HostPVelocity1, 0, memdouble);
        copyArrayFromDevice(HostPVelocity1, PVelocity1, memdouble);
		
		
		
		HostInternalDeviation = new int[Np * BlockParam];
		memset(HostInternalDeviation,0,memint);
		copyArrayFromDevice(HostInternalDeviation,InternalDeviation,memint);
							
		HostCtTime = new double[Np * BlockParam];
		memset(HostCtTime,0,memdouble);
		copyArrayFromDevice(HostCtTime,CtTime,memdouble);
		
		HostBgamma = new float[Np * BlockParam];
		memset(HostBgamma,0,memfloat);
		copyArrayFromDevice(HostBgamma,Bgamma,memfloat);
		
		HostBTimeContact = new long int[Np * BlockParam];
		memset(HostBTimeContact,0,memlongint);
		copyArrayFromDevice(HostBTimeContact,BTimeContact,memlongint);
		
		Hostgamma = new float[Np * Np * BlockParam];
		memset(Hostgamma,0,memfloat);
		copyArrayFromDevice(Hostgamma,gamma,memfloat);
		
		HostTimeInContact = new long int[Np * Np * BlockParam];
		memset(HostTimeInContact,0,memMatlongint);
		copyArrayFromDevice(HostTimeInContact,TimeInContact,memMatlongint);
		
		HostInternalClock = new int[Np * BlockParam];
		memset(HostInternalClock,0,memint);
		copyArrayFromDevice(HostInternalClock,InternalClock,memint);
		
		HostCtTimeStep = new float[Np * BlockParam];
		memset(HostCtTimeStep,0,memfloat);
		copyArrayFromDevice(HostCtTimeStep,CtTimeStep,memfloat);
		
		HostDtTimeStep = new float[Np * BlockParam];
		memset(HostDtTimeStep,0,memfloat);
		copyArrayFromDevice(HostDtTimeStep,DtTimeStep,memfloat);
		
		HostDAngRotate = new double[Np * BlockParam];
		memset(HostDAngRotate,0,memdouble);
		copyArrayFromDevice(HostDAngRotate,DAngRotate,memdouble);
		
		HostDtTime = new float[Np * BlockParam];
		memset(HostDtTime,0,memfloat);
		copyArrayFromDevice(HostDtTime,DtTime,memfloat);
		
		
		for(int g=0; g<BlockParam; g++){
			
			EightyPercentTime[g] = HostPrintOut[g]; 
			
		}

		
		ofstream zero;
		
		zero.open ("EightyPercentTime2");
		
		for(int s=0; s<BlockParam; s++){
			
			zero << HostPrintOut[s] << " ";
			
		}
		
		zero.close();
		
		
		ofstream one;
		
		one.open ("PrintOut2");
		
		for(int s=0; s<BlockParam; s++){
			
				one << HostPrintOut[s] << " ";
			
		}
		
		one.close();
		
		
		ofstream two;
		
		two.open ("ZV2");
		
		for(int s=0; s<Np * BlockParam; s++){
				
				two << HostZV[s] << " ";
				
			
		}
		
		two.close();
		
		ofstream three;
		
		three.open ("Time2");
		
		for(int s=0; s<2*50*395; s++){
			
				
				three << GetTime[s] << " ";
				
			
		}
		
		three.close();
		
		ofstream four;
		
		four.open ("Vel2");
		
		for(int s=0; s<2*50*395; s++){
				
				four << GetVel[s] << " ";
				
	
			
		}
		
		four.close();
		
		ofstream five;
		
		five.open ("Position02");
		
		for(int s=0; s<Np * BlockParam; s++){
				
				five << HostPPosition0[s] << " ";
				
			
		}
		
		five.close();
		
		ofstream six;
		
		six.open ("Position12");
		
		for(int s=0; s<Np * BlockParam; s++){
			
				
				six << HostPPosition1[s] << " ";
				
			
		}
		
		six.close();
		
		ofstream seven;
		
		seven.open ("PVelocity02");
		
		for(int s=0; s<Np * BlockParam; s++){
			
				
				seven << HostPVelocity0[s] << " ";
				
			
		}
		
		seven.close();
		
		ofstream eight;
		
		eight.open ("PVelocity12");
		
		for(int s=0; s<Np * BlockParam; s++){
			
				
				eight << HostPVelocity1[s] << " ";
				
			
		}
		
		eight.close();
		
		
		
		ofstream nine;
		
		nine.open ("InternalDeviation2");
		
		for(int s=0; s<Np * BlockParam; s++){
			
			
			nine << HostInternalDeviation[s] << " ";
			
			
		}
		
		nine.close();
		
		
		
		ofstream ten;
		
		ten.open ("CtTime2");
		
		for(int s=0; s<Np * BlockParam; s++){
			
			
			ten << HostCtTime[s] << " ";
			
			
		}
		
		ten.close();
		
		
		
		ofstream eleven;
		
		eleven.open ("Bgamma2");
		
		for(int s=0; s<Np * BlockParam; s++){
			
			
			eleven << HostBgamma[s] << " ";
			
			
		}
		
		eleven.close();
		
		
		
		ofstream twelve;
		
		twelve.open ("BTimeContact2");
		
		for(int s=0; s<Np * BlockParam; s++){
			
			
			twelve << HostBTimeContact[s] << " ";
			
			
		}
		
		twelve.close();
		
		
		
		ofstream thirteen;
		
		thirteen.open ("gamma2");
		
		for(int s=0; s<Np * Np * BlockParam; s++){
			
			
			thirteen << Hostgamma[s] << " ";
			
			
		}
		
		thirteen.close();
		
		
		
		ofstream fourteen;
		
		fourteen.open ("TimeInContact2");
		
		for(int s=0; s<Np * Np * BlockParam; s++){
			
			
			fourteen << HostTimeInContact[s] << " ";
			
			
		}
		
		fourteen.close();
		
		
		
		ofstream fifteen;
		
		fifteen.open ("InternalClock2");
		
		for(int s=0; s<Np * BlockParam; s++){
			
			
			fifteen << HostInternalClock[s] << " ";
			
			
		}
		
		fifteen.close();
		
		
		
		ofstream sixteen;
		
		sixteen.open ("CtTimeStep2");
		
		for(int s=0; s<Np * BlockParam; s++){
			
			
			sixteen << HostCtTimeStep[s] << " ";
			
			
		}
		
		sixteen.close();
		
		
		
		ofstream seventeen;
		
		seventeen.open ("DtTimeStep2");
		
		for(int s=0; s<Np * BlockParam; s++){
			
			
			seventeen << HostDtTimeStep[s] << " ";
			
			
		}
		
		seventeen.close();
		
		
		ofstream eighteen;
		
		eighteen.open ("DAngRotate2");
		
		for(int s=0; s<Np * BlockParam; s++){
			
			
			eighteen << HostDAngRotate[s] << " ";
			
			
		}
		
		eighteen.close();
		
		
		
		ofstream nineteen;
		
		nineteen.open ("DtTime2");
		
		for(int s=0; s<Np * BlockParam; s++){
			
			
			nineteen << HostDtTime[s] << " ";
			
			
		}
		
		nineteen.close();
		
		
		fstream myfileData; 
		
		char DatB[20];
		int Dn;
		Dn=sprintf(DatB,"particle_data.dat");
		//std::cout << "DatB:" << DatB << std::endl;
		
		myfileData.open(DatB,ios::out);
		
		if(myfileData.is_open()){
			
			myfileData << Np << '\n';
			
			for(int j=0; j<50; j++){
				
				if(j==50-1){
					
					myfileData << 20.0e-6 << '\n';
					
				}
				else{
					
					myfileData << 20.0e-6 << ' ';
				}
				
			}
			
			myfileData << 0 << ' ';
			
			for(int j=0; j<50; j++){
				
				myfileData << GetTime[0*2*50 + j] << ' ';
				
			}
			
			for(int j=0; j<50; j++){
				
				myfileData << GetTime[0*2*50 + 50 + j] << ' ';
				
			}
			
			for(int j=0; j<50; j++){
				
				if(j==50-1){
					
					myfileData << 0.0 << ' ';
				}
				else{
					myfileData << 0.0 << ' ';
				}
			}
			
			
			for(int j=0; j<50; j++){
				
				myfileData << 0.0 << ' ';
				
			}
			
			for(int j=0; j<50; j++){
				
				myfileData << 0.0 << ' ';
				
			}
			
			for(int j=0; j<50; j++){
				
				if(j==50-1){
					
					myfileData << 0.0 << ' ';
				}
				else{
					myfileData << 0.0 << ' ';
				}
			}
			
			
			
			for(int j=0; j<50; j++){
				
				myfileData << 0.0 << ' ';
				
			}
			
			for(int j=0; j<50; j++){
				
				myfileData << 0.0 << ' ';
				
			}
			
			for(int j=0; j<50; j++){
				
				if(j==50-1){
					
					myfileData << 0.0 << ' ';
				}
				else{
					
					myfileData << 0.0 << ' ';
					
				}
				
			}
			
			for(int Time=1; Time<395; Time++){
				
				double Vxx = 0; 
				double Vyy = 0;			
				
				myfileData << '\n' << Time << ' ';
				
				for(int j=0; j<50; j++){
					
					myfileData << GetTime[Time*2*50 + j] << ' ';
					
				}
				
				for(int j=0; j<Np; j++){
					
					myfileData << GetTime[Time*2*Np + Np + j] << ' ';
					
				}
				
				for(int j=0; j<Np; j++){
					
					if(j==Np-1){
						
						myfileData << 0.0 << ' ';
					}
					else{
						myfileData << 0.0 << ' ';
					}
				}
				
				
				for(int j=0; j<Np; j++){
					
					myfileData << GetVel[Time*2*50 + j] << ' ';
					
				}
				
				for(int j=0; j<Np; j++){
					
					myfileData << GetVel[Time*2*Np + Np + j] << ' ';
					
				}
				
				for(int j=0; j<Np; j++){
					
					if(j==Np-1){
						
						myfileData << 0.0 << ' ';
					}
					else{
						myfileData << 0.0 << ' ';
					}
				}
				double VMagg = 0;			
				for(int q=0; q<Np; q++){
					
					if((GetVel[Time*2*50 + q]==0)&&(GetVel[Time*2*Np + Np + q]==0)){
						
						Vxx += 0;
						Vyy += 0;
						
						
					}
					else{			
						VMagg = sqrt((pow(GetVel[Time*2*50 + q],2))+(pow(GetVel[Time*2*Np + Np + q],2)));
						
						Vxx += GetVel[Time*2*50 + q]/VMagg;
						Vyy += GetVel[Time*2*Np + Np + q]/VMagg;
						
					}
				}
				
				double ModAvV = sqrt((pow(Vxx,2))+(pow(Vyy,2)));			
				
				for(int j=0; j<Np; j++){
					
					double Ang = 0;
					if((GetVel[Time*2*50 + j]==0)&&(GetVel[Time*2*Np + Np + j]==0)){
						Ang = 0;
						
					}	
					else{
						double ModV = sqrt((pow(GetVel[Time*2*50 + j],2))+(pow(GetVel[Time*2*Np + Np + j],2)));
						double DotVandAv = GetVel[Time*2*50 + j]*Vxx + GetVel[Time*2*Np + Np + j]*Vyy;
						Ang = acos((DotVandAv)/(ModAvV*ModV));
						
					}
					myfileData << Ang << ' ';
					
				}
				
				for(int j=0; j<Np; j++){
					
					double Ang = 0;
					if((GetVel[Time*2*50 + j]==0)&&(GetVel[Time*2*Np + Np + j]==0)){
						Ang = 0;
						
					}
					else{
						double ModV = sqrt((pow(GetVel[Time*2*50 + j],2))+(pow(GetVel[Time*2*Np + Np + j],2)));
						double DotVandAv = GetVel[Time*2*50 + j]*Vxx + GetVel[Time*2*Np + Np + j]*Vyy;
						Ang = acos((DotVandAv)/(ModAvV*ModV));
						
					}
					
					myfileData << Ang << ' ';
					
				}
				
				for(int j=0; j<Np; j++){
					
					double Ang = 0;
					if((GetVel[Time*2*50 + j]==0)&&(GetVel[Time*2*Np + Np + j]==0)){
						Ang = 0;
						
					}
					else{
						double ModV = sqrt((pow(GetVel[Time*2*50 + j],2))+(pow(GetVel[Time*2*Np + Np + j],2)));
						double DotVandAv = GetVel[Time*2*50 + j]*Vxx + GetVel[Time*2*Np + Np + j]*Vyy;
						Ang = acos((DotVandAv)/(ModAvV*ModV));
						
					}
					
					
					if(j==Np-1){
						
						myfileData << Ang << ' ';
					}
					else{
						
						myfileData << Ang << ' ';
						
					}
					
				}
				
			}
			
			myfileData.close();
			
		}
		
		
		fstream myfileDataTime; 
		
		char DatT[20];
		int DnT;
		DnT=sprintf(DatT,"particle_times.txt");
		
		myfileDataTime.open(DatT,ios::out);
		
		if(myfileDataTime.is_open()){
			
			
			for(int g=0; g<10; g++){
				
				myfileDataTime << HostPrintOut[g] << ' ';
				
			}
			
			myfileDataTime.close();
			
		}
		
		
	}
	
	else if(runNo==2){
		
		EightyPercentTime = new float[BlockParam];
		memset(EightyPercentTime, 0, sizeof(float) * BlockParam);
		
		ifstream myEightyPercentTime;              
		
		char place0[100];
		
		int n0;
		
		n0 = sprintf(place0,"EightyPercentTime%d",runNo);  
		
		myEightyPercentTime.open(place0);
		
		string line0;
		
		if (myEightyPercentTime.is_open())
		{
			int counter = 0;
			while ( myEightyPercentTime.good() )
			{
				
				// Getline, delimited by ' '
				// Note that string use " and that char use ' 
				
				getline (myEightyPercentTime,line0,' ');
				
				// Convert String to Char 
				// Same method as above 
				
				char* LineChar =new char[line0.size()+1];
				
				LineChar[line0.size()]=0;
				
				memcpy(LineChar,line0.c_str(),line0.size());                        
				
				// Convert to double
				
				double value = strtod(LineChar,NULL);
				
				if(counter<BlockParam){
					EightyPercentTime[counter] = (float)(value);
				}
				
				counter += 1;
				
			}
			
			myEightyPercentTime.close();
			
		}


	int numThreads = 50;
	int numBlocks = 10;

	int memdouble = sizeof(double) * Np * BlockParam;
	int memfloat = sizeof(float) * Np * BlockParam;
	int memOutput = sizeof(double) * BlockParam;
	int memOutputI = sizeof(int) * BlockParam;
	int memMatdouble = sizeof(float) * (Np * Np * BlockParam);
	int memMatint = sizeof(int) * (Np * Np * BlockParam);
	int memMatfloat = sizeof(float) * (Np * Np * BlockParam);
	int memOutputEighty = sizeof(int) * Np *  BlockParam;

	int memOutputF = sizeof(float) * BlockParam;
	int memF = sizeof(float) * Np * BlockParam;
	int memint = sizeof(int) * Np * BlockParam;
	int memlongint = sizeof(long int) * Np * BlockParam;
	int memMatlongint = sizeof(long int) * (Np * Np * BlockParam);
	HostPrintOut = new float[BlockParam];
	memset(HostPrintOut, 0, memOutputF);
	
	copyArrayFromDevice(HostPrintOut,PrintOut,memOutputF);	
	
	int memSizeOut = sizeof(double) * 2*Np*395;
	
	GetTime = new double[2*50*395];
	memset(GetTime, 0, memSizeOut);
	
	copyArrayFromDevice(GetTime, TimeArray, memSizeOut);
	
	GetVel = new double[2*50*395];
	memset(GetVel, 0, memSizeOut);
	
	copyArrayFromDevice(GetVel, VelArray, memSizeOut);	

	HostZV = new int[Np * BlockParam];
	memset(HostZV, 0, memint);

			copyArrayFromDevice(HostZV,ZV,memint);

			HostPPosition0 = new double[Np * BlockParam];
			memset(HostPPosition0, 0, memdouble);
			copyArrayFromDevice(HostPPosition0, PPosition0, memdouble);
		 
			HostPPosition1 = new double[Np * BlockParam];
			memset(HostPPosition1, 0, memdouble);
			copyArrayFromDevice(HostPPosition1, PPosition1, memdouble);
		
			HostPVelocity0 = new double[Np * BlockParam];
			memset(HostPVelocity0, 0, memdouble);
			copyArrayFromDevice(HostPVelocity0, PVelocity0, memdouble);
		
			HostPVelocity1 = new double[Np * BlockParam];
			memset(HostPVelocity1, 0, memdouble);
			copyArrayFromDevice(HostPVelocity1, PVelocity1, memdouble);

			
			HostInternalDeviation = new int[Np * BlockParam];
			memset(HostInternalDeviation,0,memint);
			copyArrayFromDevice(HostInternalDeviation,InternalDeviation,memint);
			
			HostCtTime = new double[Np * BlockParam];
			memset(HostCtTime,0,memdouble);
			copyArrayFromDevice(HostCtTime,CtTime,memdouble);
			
			HostBgamma = new float[Np * BlockParam];
			memset(HostBgamma,0,memfloat);
			copyArrayFromDevice(HostBgamma,Bgamma,memfloat);
			
			HostBTimeContact = new long int[Np * BlockParam];
			memset(HostBTimeContact,0,memlongint);
			copyArrayFromDevice(HostBTimeContact,BTimeContact,memlongint);
			
			Hostgamma = new float[Np * Np * BlockParam];
			memset(Hostgamma,0,memfloat);
			copyArrayFromDevice(Hostgamma,gamma,memfloat);
			
			HostTimeInContact = new long int[Np * Np * BlockParam];
			memset(HostTimeInContact,0,memMatlongint);
			copyArrayFromDevice(HostTimeInContact,TimeInContact,memMatlongint);
			
			HostInternalClock = new int[Np * BlockParam];
			memset(HostInternalClock,0,memint);
			copyArrayFromDevice(HostInternalClock,InternalClock,memint);
			
			HostCtTimeStep = new float[Np * BlockParam];
			memset(HostCtTimeStep,0,memfloat);
			copyArrayFromDevice(HostCtTimeStep,CtTimeStep,memfloat);
			
			HostDtTimeStep = new float[Np * BlockParam];
			memset(HostDtTimeStep,0,memfloat);
			copyArrayFromDevice(HostDtTimeStep,DtTimeStep,memfloat);
			
			HostDAngRotate = new double[Np * BlockParam];
			memset(HostDAngRotate,0,memdouble);
			copyArrayFromDevice(HostDAngRotate,DAngRotate,memdouble);
			
			HostDtTime = new float[Np * BlockParam];
			memset(HostDtTime,0,memfloat);
			copyArrayFromDevice(HostDtTime,DtTime,memfloat);


			for(int g=0; g<BlockParam; g++){
                        
                                if(EightyPercentTime[g]==0){
                 
                                EightyPercentTime[g] = HostPrintOut[g];
                        
                                }
        
                        }

                        ofstream zero;
                        
                        zero.open ("EightyPercentTime2");
                        
                        for(int s=0; s<BlockParam; s++){
                        
                         	zero << HostPrintOut[s] << " ";
                        
                        }
        
                        zero.close();

                        
                        ofstream one;
                        
                        one.open ("PrintOut2");
                        
                        for(int s=0; s<BlockParam; s++){
                        
                                one << HostPrintOut[s] << " ";
                         
                        }
                        
                        one.close();


		ofstream two;
                         
                        two.open ("ZV2");
                 	        
                        for(int s=0; s<Np * BlockParam; s++){
        
                                two << HostZV[s] << " ";
                         
                        
                        }
                 
                        two.close();
                
                        ofstream three;
        
                        three.open ("Time2");
                        
                        for(int s=0; s<2*50*395; s++){
                                
                        
                                three << GetTime[s] << " ";
                         
                        
                        }
                        
                        three.close();

			ofstream four;
			four.open ("Vel2");

                        for(int s=0; s<2*50*395; s++){

                                four << GetVel[s] << " ";
                        
                        
                                
                        }
                         
                        four.close();


			ofstream five;
        
                        five.open ("Position02");
                         
                        for(int s=0; s<Np * BlockParam; s++){
                                
                                five << HostPPosition0[s] << " ";
        
                         
                        }
                        
                        five.close();


			ofstream six;

                        six.open ("Position12");

                        for(int s=0; s<Np * BlockParam; s++){


                                six << HostPPosition1[s] << " ";


                        }

                        six.close();

                        ofstream seven;

                        seven.open ("PVelocity02");

                        for(int s=0; s<Np * BlockParam; s++){


                                seven << HostPVelocity0[s] << " ";
                                
                         
                        }
                        
                        seven.close();
        
                        ofstream eight;
        
                        eight.open ("PVelocity12");
        
                        for(int s=0; s<Np * BlockParam; s++){
                         
                        
                                eight << HostPVelocity1[s] << " ";
                                
        
                        }
                         
                        eight.close();
			
			
			ofstream nine;
			
			nine.open ("InternalDeviation2");
			
			for(int s=0; s<Np * BlockParam; s++){
				
				
				nine << HostInternalDeviation[s] << " ";
				
				
			}
			
			nine.close();
			
			
			
			ofstream ten;
			
			ten.open ("CtTime2");
			
			for(int s=0; s<Np * BlockParam; s++){
				
				
				ten << HostCtTime[s] << " ";
				
				
			}
			
			ten.close();
			
			
			
			ofstream eleven;
			
			eleven.open ("Bgamma2");
			
			for(int s=0; s<Np * BlockParam; s++){
				
				
				eleven << HostBgamma[s] << " ";
				
				
			}
			
			eleven.close();
			
			
			
			ofstream twelve;
			
			twelve.open ("BTimeContact2");
			
			for(int s=0; s<Np * BlockParam; s++){
				
				
				twelve << HostBTimeContact[s] << " ";
				
				
			}
			
			twelve.close();
			
			
			
			ofstream thirteen;
			
			thirteen.open ("gamma2");
			
			for(int s=0; s<Np * Np * BlockParam; s++){
				
				
				thirteen << Hostgamma[s] << " ";
				
				
			}
			
			thirteen.close();
			
			
			
			ofstream fourteen;
			
			fourteen.open ("TimeInContact2");
			
			for(int s=0; s<Np * Np * BlockParam; s++){
				
				
				fourteen << HostTimeInContact[s] << " ";
				
				
			}
			
			fourteen.close();
			
			
			
			ofstream fifteen;
			
			fifteen.open ("InternalClock2");
			
			for(int s=0; s<Np * BlockParam; s++){
				
				
				fifteen << HostInternalClock[s] << " ";
				
				
			}
			
			fifteen.close();
			
			
			
			ofstream sixteen;
			
			sixteen.open ("CtTimeStep2");
			
			for(int s=0; s<Np * BlockParam; s++){
				
				
				sixteen << HostCtTimeStep[s] << " ";
				
				
			}
			
			sixteen.close();
			
			
			
			ofstream seventeen;
			
			seventeen.open ("DtTimeStep2");
			
			for(int s=0; s<Np * BlockParam; s++){
				
				
				seventeen << HostDtTimeStep[s] << " ";
				
				
			}
			
			seventeen.close();
			
			
			ofstream eighteen;
			
			eighteen.open ("DAngRotate2");
			
			for(int s=0; s<Np * BlockParam; s++){
				
				
				eighteen << HostDAngRotate[s] << " ";
				
				
			}
			
			eighteen.close();
			
			
			
			ofstream nineteen;
			
			nineteen.open ("DtTime2");
			
			for(int s=0; s<Np * BlockParam; s++){
				
				
				nineteen << HostDtTime[s] << " ";
				
				
			}
			
			nineteen.close();


	

	fstream myfileData; 
	
	char DatB[20];
	int Dn;
    Dn=sprintf(DatB,"particle_data.dat");
	//std::cout << "DatB:" << DatB << std::endl;
	
    myfileData.open(DatB,ios::out);
	
	if(myfileData.is_open()){
		
		myfileData << Np << '\n';
		
		for(int j=0; j<50; j++){
			
			if(j==50-1){
				
				myfileData << 20.0e-6 << '\n';
				
			}
			else{
				
				myfileData << 20.0e-6 << ' ';
			}
			
		}
		
		myfileData << 0 << ' ';
		
		for(int j=0; j<50; j++){
			
			myfileData << GetTime[0*2*50 + j] << ' ';
			
		}
		
		for(int j=0; j<50; j++){
			
			myfileData << GetTime[0*2*50 + 50 + j] << ' ';
			
		}
		
		for(int j=0; j<50; j++){
			
			if(j==50-1){
				
				myfileData << 0.0 << ' ';
			}
			else{
				myfileData << 0.0 << ' ';
			}
		}
		
		
		for(int j=0; j<50; j++){
			
			myfileData << 0.0 << ' ';
			
		}
		
		for(int j=0; j<50; j++){
			
			myfileData << 0.0 << ' ';
			
		}
		
		for(int j=0; j<50; j++){
			
			if(j==50-1){
				
				myfileData << 0.0 << ' ';
			}
			else{
				myfileData << 0.0 << ' ';
			}
		}
		
		
		
		for(int j=0; j<50; j++){
			
			myfileData << 0.0 << ' ';
			
		}
		
		for(int j=0; j<50; j++){
			
			myfileData << 0.0 << ' ';
			
		}
		
		for(int j=0; j<50; j++){
			
			if(j==50-1){
				
				myfileData << 0.0 << ' ';
			}
			else{
				
				myfileData << 0.0 << ' ';
				
			}
			
		}
		
		for(int Time=1; Time<395; Time++){

			double Vxx = 0; 
			double Vyy = 0;			

			myfileData << '\n' << Time << ' ';
			
			for(int j=0; j<50; j++){
				
				myfileData << GetTime[Time*2*50 + j] << ' ';
				
			}
			
			for(int j=0; j<Np; j++){
				
				myfileData << GetTime[Time*2*Np + Np + j] << ' ';
				
			}
			
			for(int j=0; j<Np; j++){
				
				if(j==Np-1){
					
					myfileData << 0.0 << ' ';
				}
				else{
					myfileData << 0.0 << ' ';
				}
			}
			
			
			for(int j=0; j<Np; j++){
				
				myfileData << GetVel[Time*2*50 + j] << ' ';
				
			}
			
			for(int j=0; j<Np; j++){
				
				myfileData << GetVel[Time*2*Np + Np + j] << ' ';
				
			}
			
			for(int j=0; j<Np; j++){
				
				if(j==Np-1){
					
					myfileData << 0.0 << ' ';
				}
				else{
					myfileData << 0.0 << ' ';
				}
			}
			double VMagg = 0;			
			for(int q=0; q<Np; q++){
			
			if((GetVel[Time*2*50 + q]==0)&&(GetVel[Time*2*Np + Np + q]==0)){
				
				Vxx += 0;
                                Vyy += 0;

			
			}
			else{			
			VMagg = sqrt((pow(GetVel[Time*2*50 + q],2))+(pow(GetVel[Time*2*Np + Np + q],2)));
				
				Vxx += GetVel[Time*2*50 + q]/VMagg;
				Vyy += GetVel[Time*2*Np + Np + q]/VMagg;

			}
			}
			
			double ModAvV = sqrt((pow(Vxx,2))+(pow(Vyy,2)));			

			for(int j=0; j<Np; j++){
				
				double Ang = 0;
				if((GetVel[Time*2*50 + j]==0)&&(GetVel[Time*2*Np + Np + j]==0)){
				Ang = 0;

				}	
				else{
				double ModV = sqrt((pow(GetVel[Time*2*50 + j],2))+(pow(GetVel[Time*2*Np + Np + j],2)));
				double DotVandAv = GetVel[Time*2*50 + j]*Vxx + GetVel[Time*2*Np + Np + j]*Vyy;
				Ang = acos((DotVandAv)/(ModAvV*ModV));
				
				}
				myfileData << Ang << ' ';
				
			}
			
			for(int j=0; j<Np; j++){
				
				double Ang = 0;
                                if((GetVel[Time*2*50 + j]==0)&&(GetVel[Time*2*Np + Np + j]==0)){
                                Ang = 0;

                                }
                                else{
                                double ModV = sqrt((pow(GetVel[Time*2*50 + j],2))+(pow(GetVel[Time*2*Np + Np + j],2)));
                                double DotVandAv = GetVel[Time*2*50 + j]*Vxx + GetVel[Time*2*Np + Np + j]*Vyy;
                                Ang = acos((DotVandAv)/(ModAvV*ModV));

                                }

				myfileData << Ang << ' ';
						
			}
			
			for(int j=0; j<Np; j++){

				double Ang = 0;
                                if((GetVel[Time*2*50 + j]==0)&&(GetVel[Time*2*Np + Np + j]==0)){
                                Ang = 0;

                                }
                                else{
                                double ModV = sqrt((pow(GetVel[Time*2*50 + j],2))+(pow(GetVel[Time*2*Np + Np + j],2)));
                                double DotVandAv = GetVel[Time*2*50 + j]*Vxx + GetVel[Time*2*Np + Np + j]*Vyy;
                                Ang = acos((DotVandAv)/(ModAvV*ModV));

                                }
				

				if(j==Np-1){
					
					myfileData << Ang << ' ';
				}
				else{
					
					myfileData << Ang << ' ';
					
				}
				
			}
			
		}
		
		myfileData.close();
		
	}

	
	fstream myfileDataTime; 
	
	char DatT[20];
	int DnT;
    DnT=sprintf(DatT,"particle_times.txt");
	
	myfileDataTime.open(DatT,ios::out);
	
	if(myfileDataTime.is_open()){
		
		
		for(int g=0; g<10; g++){
			
		myfileDataTime << HostPrintOut[g] << ' ';
		
		}
		
		myfileDataTime.close();
		
	}
	


		
	}

}

