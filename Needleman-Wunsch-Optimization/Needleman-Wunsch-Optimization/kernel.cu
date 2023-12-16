/************************************************************************************************
*						This code is written by Mohammed Ali Alawi Shehab
*						Publication name :Speed Up Needleman-Wunsch Global Alignment Algorithm Using GPU Technique
*						URL				: https://www.researchgate.net/publication/292977570_Speed_Up_Needleman-Wunsch_Global_Alignment_Algorithm_Using_GPU_Technique#feedback/198672
*						Authors			:  Maged Fakirah,  Mohammed A. Shehab,  Yaser Jararweh and Mahmoud Al-Ayyoub
*						INSTITUTION		: Jordan University of Science and Technology, Irbid, Jordan
*						DEPARTMENT		: Department of Computer Science
*************************************************************************************************/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#define MIN(x,y) ((x) < (y) ? (x) : (y))

__global__ void ShihabKernel(char* Adata, char* Bdata, int slice, int z, int blen, int* NewH, int Increment, int Max)
{
	//int i = threadIdx.x;

	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int i = ((gridDim.x * blockDim.x) * y) + x;

	if (i <= Max)
	{
		int match = 0;
		int mismatch = 1;

		int startIndex;
		if (z <= 0)
		{
			startIndex = slice;
		}
		else
		{
			startIndex = Increment * z + slice;
		}

		int j = startIndex + (i * Increment);

		int row = j / blen;
		int column = j % blen;

		if (row == 0 || column == 0)
		{
			return;
		}

		int score = (Adata[row - 1] == Bdata[column - 1]) ? match : mismatch;
		NewH[column + row * blen] = MIN(NewH[(column - 1) + (row - 1) * blen] + score, MIN(NewH[(column)+(row - 1) * blen] + 1, NewH[(column - 1) + (row)*blen] + 1));
	}
}

__global__ void init_rows(int* NewH, int blen)
{
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int i = ((gridDim.x * blockDim.x) * y) + x;


	int row = i / blen;
	int column = i % blen;

	if (row == 0 && column > 0)
	{
		NewH[column + row * blen] = i;
	}
}

__global__ void init_columns(int* NewH, int blen)
{
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int i = ((gridDim.x * blockDim.x) * y) + x;


	int row = i / blen;
	int column = i % blen;

	if (column == 0 && row > 0)
	{
		NewH[column + row * blen] = row;
	}
}

char GetNewChar(int num)
{
	char c = 'A';

	switch (num)
	{
	case 0: c = 'A'; break;
	case 1: c = 'C'; break;
	case 2: c = 'G'; break;
	case 3: c = 'T'; break;
	}
	return c;
}

int main(int argc, char* argv[])
{
	char* a;
	char* b;
	clock_t CPUbegin, CPUend;
	clock_t GPUbegin, GPUend;
	long double time_spentCPU, time_spentGPU;
	long int LENGHT = 0;

	printf("Enter the lenght of string: ");
	scanf("%ld", &LENGHT);

	a = (char*)malloc(LENGHT * sizeof(char));
	b = (char*)malloc(LENGHT * sizeof(char));
	long int t;

	for (t = 0; t < LENGHT; ++t)
	{
		//This loop for initialize Random two sequences DNA
		int num = rand() % 4;

		a[t] = GetNewChar(num);
		num = rand() % 4;
		b[t] = GetNewChar(num);
	}
	// Write end of string to stop reading process
	a[t] = '\0';
	b[t] = '\0';

	int i, j;
	int score;

	int alen = strlen(a) + 1;
	int blen = strlen(b) + 1;
	int* NewH;
	int* H;


	NewH = (int*)malloc(alen * blen * sizeof(int));
	H = (int*)malloc(alen * blen * sizeof(int));

	//------------------Initializing The Matricies-------------------

	int* dev_H = 0;
	char* dev_a;
	char* dev_b;

	NewH[0] = 0;
	H[0] = 0;

	CPUbegin = clock();										//begain time of CPU
	//Initilize the first row and columns
	for (i = 1; i < blen; ++i)
	{
		H[i] = i;
	}

	for (j = 1; j < alen; j++)
	{
		H[blen * j] = j;
	}

	//---------------------Filling The Matricies----------------------
	// The Diagonal technique CPU section

	for (int slice = 0; slice < 2 * alen - 1; ++slice)
	{
		int z = slice < alen ? 0 : slice - alen + 1;
		for (int j = z; j <= slice - z; ++j)
		{
			int row = j;
			int column = (slice - j);

			if (row == 0 || column == 0)
			{
				continue;
			}
			score = (a[row - 1] == b[column - 1]) ? 0 : 1;
			H[(column)+row * blen] = MIN(H[(column - 1) + (row - 1) * blen] + score, MIN(H[(column)+(row - 1) * blen] + 1, H[(column - 1) + (row)*blen] + 1));

		}
	}

	CPUend = clock();										//End time of CPU
	time_spentCPU = (double)(CPUend - CPUbegin) / CLOCKS_PER_SEC;
	printf("CPU time = %lf Sec\n", time_spentCPU);


	//Here This Block of code is to Print data after finishing code

	//printf("\n____________CPU_______________\n");

	// for(int r = 0 ; r < alen ; r++)   
	// {
	//   for(int c = 0 ; c < blen ; c++)
	//{
	//	 //printf("Type a number for <line: %d, column: %d>\t", i, j);
	//	printf("%3d ", H[r *blen +c]);// printf("\n");
	//}
	//     printf("\n");
	// }



	cudaSetDevice(0);

	//Create memory allocation in GPU
	cudaMalloc((void**)&dev_H, alen * blen * sizeof(int));
	cudaMalloc((void**)&dev_a, LENGHT * sizeof(char));
	cudaMalloc((void**)&dev_b, LENGHT * sizeof(char));

	GPUbegin = clock();
	//Copy all arrays to GPU memory
	cudaMemcpy(dev_H, NewH, alen * blen * sizeof(int), cudaMemcpyHostToDevice);

	const int NumberOfThreads = 256;
	dim3 ThreadsWarp(32, 8);
	// Here run initionlize in GPU side

	init_rows <<<ThreadsWarp, alen >> > (dev_H, blen);
	init_columns <<<ThreadsWarp, blen >> > (dev_H, blen);


	//cudaMemcpy(H, dev_H,  alen * blen  * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(dev_a, a, LENGHT * sizeof(char), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, LENGHT * sizeof(char), cudaMemcpyHostToDevice);


	//Set GPU for parallel working

	int size = (int)ceil((float)blen / (float)NumberOfThreads);
	int Increment = alen - 1;
	//begain time of GPU
	int MemSize = alen * blen;
	for (int slice = 0; slice < 2 * alen - 1; ++slice)
	{
		int z = slice < alen ? 0 : slice - alen + 1;//CPU
		size = (int)ceil((float)((slice - 2 * z) + 1));

		ShihabKernel <<<ThreadsWarp, MemSize >> > (dev_a, dev_b, slice, z, alen, dev_H, Increment, size);
	}

	cudaMemcpy(NewH, dev_H, alen * blen * sizeof(int), cudaMemcpyDeviceToHost);
	GPUend = clock();								//End time of GPU

	time_spentGPU = (double)(GPUend - GPUbegin) / CLOCKS_PER_SEC;
	printf("\nGPU time = %lf Sec\n", time_spentGPU);

	/*cudaMemcpy(a, dev_a,  LENGHT * sizeof(char), cudaMemcpyDeviceToHost);
	cudaMemcpy(b, dev_b, LENGHT  * sizeof(char), cudaMemcpyDeviceToHost);
	printf("a = %s\nb = %s\n",a,b);*/
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_H);
	cudaDeviceReset();


	//---------------------printing the matricies---------------------


	// printf("\n____________GPU_______________\n");

	// for(int r = 0 ; r < alen ; r++)   
	// {
	//   for(int c = 0 ; c < blen ; c++)
	//{
	//	// printf("Type a number for <line: %d, column: %d>\t", i, j);
	//	printf("%3d ", NewH[r *blen +c]);// printf("\n");
	//}
	//     printf("\n");
	// }

	return (0);
}
