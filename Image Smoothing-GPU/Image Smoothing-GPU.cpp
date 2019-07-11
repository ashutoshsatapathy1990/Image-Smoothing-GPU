
//--------------------------------------------------------------------------------------------------|
// IMAGE SMOOTHING IN FREQUENCY DOMAIN USING DISCRETE FOURIER TRANSFORMATION AND IDEAL FILTER.	    |
// IMAGE SMOOTHING IN FREQUENCY DOMAIN USING DISCRETE FOURIER TRANSFORMATION AND GAUSSIAN FILTER,   |
// IMAGE SMOOTHING IN FREQUENCY DOMAIN USING DISCRETE FOURIER TRANSFORMSTION AND BUTTERWORTH FILTER.|
// Image Smoothing.cpp : Defines the entry point for the console application.			    |
//--------------------------------------------------------------------------------------------------|

//++++++++++++++++++++++++++++++++++ START HEADER FILES +++++++++++++++++++++++++++++++++++++++++++++
// Include The Necesssary Header Files
// [Both In std. and non std. path]
#include "stdafx.h"
#include<stdio.h>
#include<conio.h>
#include<string.h>
#include<stdlib.h>
#include<complex>
#ifdef __APPLE__
#include<OpenCL\cl.h>
#else
#include<CL\cl.h>
#endif
#include<opencv\cv.h>
#include<opencv\highgui.h>
using namespace std;
using namespace cv;

//++++++++++++++++++++++++++++++++++ END HEADER FILES +++++++++++++++++++++++++++++++++++++++++++++++

//+++++++++++++++++++++++++++++++++ HORIZONTAL DFT KERNEL +++++++++++++++++++++++++++++++++++++++++++
// OpenCL Horizontal Row Wise DFT Kernel Which Is Run For Every Work Item Created.
const char* HDFT_Kernel =
"#pragma OPENCL EXTENSION cl_khr_fp32 : enable									\n"
"__kernel													\n"
"void HDFT_Kernel(__global const uchar* data,									\n"
"					__global float2* in,							\n"
"					__local float2* SharedArray,						\n"
"					int width,								\n"
"					float norm)								\n"
"{														\n"
"	size_t globalID = get_global_id(0);									\n"
"	size_t localID = get_local_id(0);									\n"
"	size_t groupID = get_group_id(0);									\n"
"	size_t groupSize = get_local_size(0);									\n"
"	int v = globalID % width;										\n"
"	float param = (-2.0*v)/width;										\n"
"	SharedArray[localID] = (0.0, 0.0);									\n"
"	float c, s;												\n"
"	int valueH;												\n"
"	// Horizontal DFT Transformation									\n"
"	for (int i = 0; i < groupSize; i++)									\n"
"	{													\n"
"		valueH = (int)data[groupSize * groupID + i];							\n"
"		s = sinpi(i * param);										\n"
"		c = cospi(i * param);										\n"
"		SharedArray[localID].x += valueH * c;								\n"
"		SharedArray[localID].y += valueH * s;								\n"
"	}													\n"
"	in[groupSize * groupID +localID].x = norm*SharedArray[localID].x;					\n"
"	in[groupSize * groupID +localID].y = norm*SharedArray[localID].y;					\n"
"}														\n"
"\n";

//++++++++++++++++++++++++++++++++++++ END H-DFT KERNEL +++++++++++++++++++++++++++++++++++++++++++++

//+++++++++++++++++++++++++++++++++ HORIZONTAL IDFT KERNEL ++++++++++++++++++++++++++++++++++++++++++
// OpenCL Horizontal Row Wise IDFT Kernel Which Is Run For Every Work Item Created.
const char* HIDFT_Kernel =
"#pragma OPENCL EXTENSION cl_khr_fp32 : enable									\n"
"__kernel													\n"
"void HIDFT_Kernel(__global uchar* data,									\n"
"					__global const float2* in,						\n"
"					__local float2* SharedArray,						\n"
"					int width,								\n"
"					float norm)								\n"
"{														\n"
"	int globalID = get_global_id(0);									\n"
"	int localID = get_local_id(0);										\n"
"	int groupID = get_group_id(0);										\n"
"	int groupSize = get_local_size(0);									\n"
"	int v = globalID % width;										\n"
"	float param = (2.0*v)/width;										\n"
"	SharedArray[localID] = (0.0, 0.0);									\n"
"	float c, s;												\n"
"	float2 value;												\n"
"	// Horizontal IDFT Transformation									\n"
"	for (int i = 0; i < groupSize; i++)									\n"
"	{													\n"
"		value = in[groupSize * groupID + i];								\n"
"		s = sinpi(i * param);										\n"
"		c = cospi(i * param);										\n"
"		SharedArray[localID].x +=	value.x * c - value.y * s;					\n"
"		SharedArray[localID].y +=	value.x * s + value.y * c;					\n"
"	}													\n"
"	data[groupSize*groupID +localID] = (uchar)(norm*SharedArray[localID].x);				\n"
"}														\n"
"\n";

//++++++++++++++++++++++++++++++++++++ END H-IDFT KERNEL ++++++++++++++++++++++++++++++++++++++++++++

//+++++++++++++++++++++++++++++++++++ VERTICAL DFT KERNEL +++++++++++++++++++++++++++++++++++++++++++
// OpenCL Vertical Column Wise DFT Kernel Which Is Run For Every Work Item Created.
const char* VDFT_Kernel =
"#pragma OPENCL EXTENSION cl_khr_fp32 : enable									\n"
"__kernel													\n"
"void VDFT_Kernel(__global const float2* in,									\n"
"					__global float2* out,							\n"
"					__local float2* SharedArray,						\n"
"					int width,								\n"
"					float norm)								\n"
"{														\n"
"	size_t globalID = get_global_id(0);									\n"
"	size_t localID = get_local_id(0);									\n"
"	size_t groupID = get_group_id(0);									\n"
"	size_t groupSize = get_local_size(0);									\n"
"	int v = globalID % width;										\n"
"	float param = (-2.0*v)/width;										\n"
"	SharedArray[localID] = (0.0, 0.0);									\n"
"	float c, s, valueH;											\n"
"	float2 value;										         	\n"
"	// Horizontal DFT Transformation									\n"
"	for (int i = 0; i < groupSize; i++)									\n"
"	{													\n"
"		value = in[groupSize * i + groupID];								\n"
"		s = sinpi(i * param);										\n"
"		c = cospi(i * param);										\n"
"		SharedArray[localID].x += value.x * c - value.y * s;						\n"
"		SharedArray[localID].y += value.x * s + value.y * c;						\n"
"	}													\n"
"	out[groupSize * localID + groupID].x= norm*SharedArray[localID].x;					\n"
"	out[groupSize * localID + groupID].y= norm*SharedArray[localID].y;					\n"
"}														\n"
"\n";

//++++++++++++++++++++++++++++++++++++ END V-DFT KERNEL +++++++++++++++++++++++++++++++++++++++++++++

//+++++++++++++++++++++++++++++++++++ VERTICAL IDFT KERNEL ++++++++++++++++++++++++++++++++++++++++++
// OpenCL Vertical Column Wise IDFT Kernel Which Is Run For Every Work Item Created.
const char* VIDFT_Kernel =
"#pragma OPENCL EXTENSION cl_khr_fp32 : enable									\n"
"__kernel													\n"
"void VIDFT_Kernel(__global float2* in,										\n"
"					__global const float2* out,						\n"
"					__local float2* SharedArray,						\n"
"					int width,								\n"
"					float norm)								\n"
"{														\n"
"	int globalID = get_global_id(0);									\n"
"	int localID = get_local_id(0);										\n"
"	int groupID = get_group_id(0);										\n"
"	int groupSize = get_local_size(0);									\n"
"	int v = globalID % width;										\n"
"	float param = (2.0*v)/width;										\n"
"	SharedArray[localID] = (0.0, 0.0);									\n"
"	float c, s;												\n"
"	float2 value;												\n"
"	// Horizontal IDFT Transformation									\n"
"	for (int i = 0; i < groupSize; i++)									\n"
"	{													\n"
"		value = out[groupSize * i + groupID];								\n"
"		s = sinpi(i * param);										\n"
"		c = cospi(i * param);										\n"
"		SharedArray[localID].x +=	value.x * c - value.y * s;					\n"
"		SharedArray[localID].y +=	value.x * s + value.y * c;					\n"
"	}													\n"
"	in[groupSize * localID + groupID].x= norm*SharedArray[localID].x;					\n"
"	in[groupSize * localID + groupID].y= norm*SharedArray[localID].y;					\n"
"}														\n"
"\n"

//++++++++++++++++++++++++++++++++++++ END V-IDFT KERNEL ++++++++++++++++++++++++++++++++++++++++++++

//+++++++++++++++++++++++++++++++++++ IDEAL KERNEL ++++++++++++++++++++++++++++++++++++++++++++++++++
// OpenCL Low Pass Ideal Filter Kernel Which Is Run For Every Work Item Created
const char *ideal_kernel =
"#define EXP 2.72												\n" \
"#pragma OPENCL EXTENSION cl_khr_fp32 : enable									\n" \
"#pragma OPENCL EXTENSION cl_khr_printf : enable								\n" \
"__kernel													\n" \
"void ideal_kernel (__global float2* data,					   				\n" \
"		int height,											\n" \
"		int width,											\n" \
"		int CUTOFF)											\n" \
"{														\n" \
"	// Get the index of work items										\n" \
"	uint index = get_global_id(0);										\n" \
"	int U = index / width;											\n" \
"	int V = index % width;											\n" \
"	float imgx = data[index].x;										\n" \
"	float imgy = data[index].y;										\n" \
"	float D = pow(height/2 - abs(U - height/2), 2.0) + pow(width/2 - abs(V - width/2), 2.0);		\n" \
"	float H = (sqrt(D) < CUTOFF)? 1.0 : 0.0;								\n" \
"	data[index].x = imgx * H;										\n" \
"	data[index].y = imgy * H;										\n" \
"}														\n" \
"\n";
//+++++++++++++++++++++++++++++++++ END IDEAL KERNEL ++++++++++++++++++++++++++++++++++++++++++++++++

//+++++++++++++++++++++++++++++++++++ GAUSSIAN KERNEL +++++++++++++++++++++++++++++++++++++++++++++++
// OpenCL Low Pass Gaussian Filter Kernel Which Is Run For Every Work Item Created
const char *gaussian_kernel =
"#define EXP 2.72												\n" \
"#pragma OPENCL EXTENSION cl_khr_fp32 : enable									\n" \
"#pragma OPENCL EXTENSION cl_khr_printf : enable								\n" \
"__kernel													\n" \
"void gaussian_kernel (__global float2* data,									\n" \
"		int height,											\n" \
"		int width,											\n" \
"		int CUTOFF)											\n" \
"{														\n" \
"	// Get the index of work items										\n" \
"	uint index = get_global_id(0);										\n" \
"	int U = index / width;											\n" \
"	int V = index % width;											\n" \
"	float imgx = data[index].x;										\n" \
"	float imgy = data[index].y;										\n" \
"	float D = pow(height/2 - abs(U - height/2), 2.0) + pow(width/2 - abs(V - width/2), 2.0);		\n" \
"	float H = pow(EXP, (-1.0 * D / (2.0 * pow(CUTOFF, 2.0))));						\n" \
"	data[index].x = imgx * H;										\n" \
"	data[index].y = imgy * H;										\n" \
"}														\n" \
"\n";
//+++++++++++++++++++++++++++++++++ END GAUSSIAN KERNEL +++++++++++++++++++++++++++++++++++++++++++++

//+++++++++++++++++++++++++++++++++++ BUTTERWORTH KERNEL ++++++++++++++++++++++++++++++++++++++++++++
// OpenCL Low Pass Gaussian Filter Kernel Which Is Run For Every Work Item Created
const char *butterworth_kernel =
"#define EXP 2.72												\n" \
"#pragma OPENCL EXTENSION cl_khr_fp32 : enable									\n" \
"#pragma OPENCL EXTENSION cl_khr_printf : enable								\n" \
"__kernel													\n" \
"void butterworth_kernel (__global float2* data,								\n" \
"		int height,											\n" \
"		int width,											\n" \
"		int CUTOFF,											\n" \
"		float Ord)											\n" \
"{														\n" \
"	// Get the index of work items										\n" \
"	uint index = get_global_id(0);										\n" \
"	int U = index / width;											\n" \
"	int V = index % width;											\n" \
"	float imgx = data[index].x;										\n" \
"	float imgy = data[index].y;										\n" \
"	float D = pow(height/2 - abs(U - height/2), 2.0) + pow(width/2 - abs(V - width/2), 2.0);		\n" \
"	float H = 1.0 / (1 + pow (sqrt(D) / CUTOFF, Ord));							\n" \
"	data[index].x = imgx * H;										\n" \
"	data[index].y = imgy * H;										\n" \
"}														\n" \
"\n";
//+++++++++++++++++++++++++++++++++ END BUTTERWORTH KERNEL ++++++++++++++++++++++++++++++++++++++++++

//+++++++++++++++++++++++++++++++++ CREATE AND BUILD PROGRAM ++++++++++++++++++++++++++++++++++++++++
// Create and Build The Program Using Selected Low Pass Filter's OpenCL Kernel Source Code.
cl_program Lowpass_Program(cl_context context, const char *name, cl_device_id* device_list, cl_int clStatus)
{
	cl_program program = clCreateProgramWithSource(context, 1, (const char **)&name, NULL, &clStatus);
	clStatus = clBuildProgram(program, 1, device_list, NULL, NULL, NULL);
	return program;
}
//++++++++++++++++++++++++++++++ END CREATE AND BUILD PROGRAM +++++++++++++++++++++++++++++++++++++++

//+++++++++++++++++++++++++++++++++ CREATE OPENCL KERNELS +++++++++++++++++++++++++++++++++++++++++++
// Create Multiple Kernels from The Program dedicated for [R, G, B] Channels of The Input Image. 
cl_kernel* Lowpass_Kernel(cl_program program, cl_kernel* kernel, char* name, cl_int clStatus)
{
	kernel[0] = clCreateKernel(program, name, &clStatus);
	kernel[1] = clCreateKernel(program, name, &clStatus);
	kernel[2] = clCreateKernel(program, name, &clStatus);
	return kernel;

}
//+++++++++++++++++++++++++++++++ END CREATE OPENCL KERNELS +++++++++++++++++++++++++++++++++++++++++

//+++++++++++++++++++++++++++++++++ SET KERNELS' ARGUMENTS ++++++++++++++++++++++++++++++++++++++++++
// Passing Arguments to Each Kernels Dedicated for Red, Green and Blue Channels [Image Smoothing]
void Kernel_Arg(cl_kernel* kernel, cl_mem RED_clmem, cl_mem GREEN_clmem, cl_mem BLUE_clmem, int rows, int cols, int CUTOFF, float Ord, int Select, cl_int clStatus)
{
	clStatus = clSetKernelArg(kernel[0], 0, sizeof(cl_mem), (void *)&RED_clmem);
	clStatus = clSetKernelArg(kernel[0], 1, sizeof(int), (void *)&rows);
	clStatus = clSetKernelArg(kernel[0], 2, sizeof(int), (void *)&cols);
	clStatus = clSetKernelArg(kernel[0], 3, sizeof(int), (void *)&CUTOFF);
	clStatus = clSetKernelArg(kernel[1], 0, sizeof(cl_mem), (void *)&GREEN_clmem);
	clStatus = clSetKernelArg(kernel[1], 1, sizeof(int), (void *)&rows);
	clStatus = clSetKernelArg(kernel[1], 2, sizeof(int), (void *)&cols);
	clStatus = clSetKernelArg(kernel[1], 3, sizeof(int), (void *)&CUTOFF);
	clStatus = clSetKernelArg(kernel[2], 0, sizeof(cl_mem), (void *)&BLUE_clmem);
	clStatus = clSetKernelArg(kernel[2], 1, sizeof(int), (void *)&rows);
	clStatus = clSetKernelArg(kernel[2], 2, sizeof(int), (void *)&cols);
	clStatus = clSetKernelArg(kernel[2], 3, sizeof(int), (void *)&CUTOFF);
	// If Butterworth Low Pass Filter, Pass The Order of The Filter.
	if (Select == 3)
	{
		clStatus = clSetKernelArg(kernel[0], 4, sizeof(float), (void *)&Ord);
		clStatus = clSetKernelArg(kernel[1], 4, sizeof(float), (void *)&Ord);
		clStatus = clSetKernelArg(kernel[2], 4, sizeof(float), (void *)&Ord);
	}
}
//+++++++++++++++++++++++++++++++ END OPENCL KERNELS' ARGUMENTS +++++++++++++++++++++++++++++++++++++

//+++++++++++++++++++++++++++++++++ SET DFT KERNELS' ARGUMENTS ++++++++++++++++++++++++++++++++++++++
// Passing Arguments to Each Kernels Dedicated for Red, Green and Blue Channels [DFT and IDFT]
void DFT_Kernel_Arg(cl_kernel* kernel, cl_mem RED_clmem, cl_mem GREEN_clmem, cl_mem BLUE_clmem, cl_mem Trans_RED_clmem, cl_mem Trans_GREEN_clmem, cl_mem Trans_BLUE_clmem, int rows, float norm, cl_int clStatus)
{
	clStatus = clSetKernelArg(kernel[0], 0, sizeof(cl_mem), (void *)&RED_clmem);
	clStatus = clSetKernelArg(kernel[0], 1, sizeof(cl_mem), (void *)&Trans_RED_clmem);
	clStatus = clSetKernelArg(kernel[0], 2, sizeof(complex<float>)*rows, NULL);
	clStatus = clSetKernelArg(kernel[0], 3, sizeof(int), (void *)&rows);
	clStatus = clSetKernelArg(kernel[0], 4, sizeof(float), (void *)&norm);
	clStatus = clSetKernelArg(kernel[1], 0, sizeof(cl_mem), (void *)&GREEN_clmem);
	clStatus = clSetKernelArg(kernel[1], 1, sizeof(cl_mem), (void *)&Trans_GREEN_clmem);
	clStatus = clSetKernelArg(kernel[1], 2, sizeof(complex<float>)*rows, NULL);
	clStatus = clSetKernelArg(kernel[1], 3, sizeof(int), (void *)&rows);
	clStatus = clSetKernelArg(kernel[1], 4, sizeof(float), (void *)&norm);
	clStatus = clSetKernelArg(kernel[2], 0, sizeof(cl_mem), (void *)&BLUE_clmem);
	clStatus = clSetKernelArg(kernel[2], 1, sizeof(cl_mem), (void *)&Trans_BLUE_clmem);
	clStatus = clSetKernelArg(kernel[2], 2, sizeof(complex<float>)*rows, NULL);
	clStatus = clSetKernelArg(kernel[2], 3, sizeof(int), (void *)&rows);
	clStatus = clSetKernelArg(kernel[2], 4, sizeof(float), (void *)&norm);
}

//+++++++++++++++++++++++++++++ END OPENCL DFT KERNELS' ARGUMENTS ++++++++++++++++++++++++++++++++++++

//+++++++++++++++++++++++++++++++++++++++ EXECUTE KERNELS +++++++++++++++++++++++++++++++++++++++++++
//Execute the OpenCL Kernels for Smoothing of Each Channels Independetly.
void Exec_Kernel(cl_command_queue command_queue, cl_kernel* kernel, size_t global, size_t local, cl_int clStatus)
{
	clStatus = clEnqueueNDRangeKernel(command_queue, kernel[0], 1, NULL, &global, &local, 0, NULL, NULL);
	clStatus = clEnqueueNDRangeKernel(command_queue, kernel[1], 1, NULL, &global, &local, 0, NULL, NULL);
	clStatus = clEnqueueNDRangeKernel(command_queue, kernel[2], 1, NULL, &global, &local, 0, NULL, NULL);
}
//+++++++++++++++++++++++++++++++++++++ END EXECUTE KERNELS +++++++++++++++++++++++++++++++++++++++++

void releaseKernel(cl_kernel* kernel, cl_int clStatus)
{
	clStatus = clReleaseKernel(kernel[0]);
	clStatus = clReleaseKernel(kernel[1]);
	clStatus = clReleaseKernel(kernel[2]);
}

//++++++++++++++++++++++++++++++++++ START MAIN PROGRAM +++++++++++++++++++++++++++++++++++++++++++++
int main()
{
	// Declare Variables for CUTOFF Frequency and Order of The Filter
	// Delcare Variables for Selection of a Filter.
	int CUTOFF, Select;
	float Ord;

	// Initialize Clock Variable to compute Time Taken in millisecs
	clock_t start, end;
	float Time_Used;

	// Enter CUTOFF Frequency and Order of The Filter
	// Enter Selection of Filter [1: Ideal, 2: Gaussian, 3: Butterworth]
	printf("ENTER THE CUTOFF FREQUENCY AND ORDER OF THE FILTER: \n");
	scanf("%d %f", &CUTOFF, &Ord);
	printf("ENTER YOUR CHOICE FOR IMAGE DENOISIFICATION. [1]IDEAL [2]GAUSSIAN [3]BUTTERWORTH :\n");
	scanf("%d", &Select);

	// Get The Platforms' Information
	cl_platform_id* platforms = NULL;
	cl_uint num_platforms;

	// Set up The Platforms
	cl_int clStatus = clGetPlatformIDs(0, NULL, &num_platforms);
	platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);
	clStatus = clGetPlatformIDs(num_platforms, platforms, NULL);

	// Get The Device Lists and Choose The Device You Want to Run on.
	cl_device_id* device_list = NULL;
	cl_uint num_devices;
	clStatus = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
	device_list = (cl_device_id*)malloc(sizeof(cl_device_id) * num_devices);
	clStatus = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, num_devices, device_list, NULL);

	// Create an OpenCL Context for Each Device in The Platform
	cl_context context;
	context = clCreateContext(NULL, num_devices, device_list, NULL, NULL, &clStatus);

	// Create a Command Queue for Out of Order Execution in 0th Device.
	cl_command_queue command_queue_smooth = clCreateCommandQueue(context, device_list[0], CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &clStatus);

	// Read Noisy Image from The Given Path
	Mat Image = imread("Noise_Gaus.bmp", CV_LOAD_IMAGE_COLOR);

	// Check The Status of Image <Mat Variable>
	if (!Image.data)
	{
		printf("COULDN'T OPEN OR READ INPUT FILE");
		return -1;
	}

	// Display The Input Noisy Image
	namedWindow("NOISY IMAGE", WINDOW_NORMAL);
	imshow("NOISY IMAGE", Image);

	// Variables to Store [R, G, B] Channels of Noisy Image
	Mat RGB_Image[3], R_Image, G_Image, B_Image;

	// Create Host Buffers for Each Channels [R, G, B] as Image Size [H x W]
	uchar* RED_Frame = (uchar*)malloc(sizeof(uchar) * Image.rows * Image.cols);
	uchar* GREEN_Frame = (uchar*)malloc(sizeof(uchar) * Image.rows * Image.cols);
	uchar* BLUE_Frame = (uchar*)malloc(sizeof(uchar) * Image.rows * Image.cols);

	// Create OpenCL Device Buffers and Map to Host Buffers Separately Created for Each Color Channel. 
	cl_mem RED_clmem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(uchar) * Image.rows * Image.cols, RED_Frame, &clStatus);
	cl_mem GREEN_clmem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(uchar) * Image.rows * Image.cols, GREEN_Frame, &clStatus);
	cl_mem BLUE_clmem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(uchar) * Image.rows * Image.cols, BLUE_Frame, &clStatus);
	cl_mem Trans_RED_clmem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(complex<float>) * Image.rows * Image.cols, NULL, &clStatus);
	cl_mem Trans_GREEN_clmem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(complex<float>) * Image.rows * Image.cols, NULL, &clStatus);
	cl_mem Trans_BLUE_clmem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(complex<float>) * Image.rows * Image.cols, NULL, &clStatus);
	cl_mem TR_temp = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(complex<float>) * Image.rows * Image.cols, NULL, &clStatus);
	cl_mem TG_temp = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(complex<float>) * Image.rows * Image.cols, NULL, &clStatus);
	cl_mem TB_temp = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(complex<float>) * Image.rows * Image.cols, NULL, &clStatus);

	// Get The Program's and Kernels' Information
	cl_program program_smooth = NULL, program_HDFT = NULL, program_HIDFT = NULL, program_VDFT = NULL, program_VIDFT = NULL;
	cl_kernel* kernel_smooth = (cl_kernel*)malloc(sizeof(cl_kernel) * 3);
	cl_kernel* kernel_HDFT = (cl_kernel*)malloc(sizeof(cl_kernel) * 3);
	cl_kernel* kernel_HIDFT = (cl_kernel*)malloc(sizeof(cl_kernel) * 3);
	cl_kernel* kernel_VDFT = (cl_kernel*)malloc(sizeof(cl_kernel) * 3);
	cl_kernel* kernel_VIDFT = (cl_kernel*)malloc(sizeof(cl_kernel) * 3);

	// Creation and Building of OpenCL DFT and IDFT Programs for Each Channels [R, G, B].
	program_HDFT = Lowpass_Program(context, HDFT_Kernel, device_list, clStatus);
	program_HIDFT = Lowpass_Program(context, HIDFT_Kernel, device_list, clStatus);
	program_VDFT = Lowpass_Program(context, VDFT_Kernel, device_list, clStatus);
	program_VIDFT = Lowpass_Program(context, VIDFT_Kernel, device_list, clStatus);

	// Creation of OpenCL HDFT, VDFT, HIDFT, VIDFT Kernels for Each Channels [R, G, B].
	kernel_HDFT = Lowpass_Kernel(program_HDFT, kernel_HDFT, "HDFT_Kernel", clStatus);
	kernel_HIDFT = Lowpass_Kernel(program_HIDFT, kernel_HIDFT, "HIDFT_Kernel", clStatus);
	kernel_VDFT = Lowpass_Kernel(program_VDFT, kernel_VDFT, "VDFT_Kernel", clStatus);
	kernel_VIDFT = Lowpass_Kernel(program_VIDFT, kernel_VIDFT, "VIDFT_Kernel", clStatus);

	// Passing Arguments to DFT and IDFT Kernels for Image Transformation Before and After Image Enhancement.
	DFT_Kernel_Arg(kernel_HDFT, RED_clmem, GREEN_clmem, BLUE_clmem, TR_temp, TG_temp, TB_temp, Image.rows, 1.0/sqrt(Image.rows), clStatus);
	DFT_Kernel_Arg(kernel_VDFT, TR_temp, TG_temp, TB_temp, Trans_RED_clmem, Trans_GREEN_clmem, Trans_BLUE_clmem, Image.rows, 1.0/sqrt(Image.rows), clStatus);
	DFT_Kernel_Arg(kernel_VIDFT, TR_temp, TG_temp, TB_temp, Trans_RED_clmem, Trans_GREEN_clmem, Trans_BLUE_clmem, Image.rows, 1.0/sqrt(Image.rows), clStatus);
	DFT_Kernel_Arg(kernel_HIDFT, RED_clmem, GREEN_clmem, BLUE_clmem, TR_temp, TG_temp, TB_temp, Image.rows, 1.0/sqrt(Image.rows), clStatus);

	// Selection of Low Pass Filter: [0]Ideal [1]Gaussian [2]Butterworth
	// Creation and Building of OpenCL Programs and Kernels for Each Channels [R, G, B].
	// Passing Arguments to Kernels for Image smoothing based on Selected Filter.
	switch (Select)
	{
	case 1: printf("YOU HAVE SELECTED IDEAL FILTER \n");
		program_smooth = Lowpass_Program(context, ideal_kernel, device_list, clStatus);
		kernel_smooth = Lowpass_Kernel(program_smooth, kernel_smooth, "ideal_kernel", clStatus);
		Kernel_Arg(kernel_smooth, Trans_RED_clmem, Trans_GREEN_clmem, Trans_BLUE_clmem, Image.rows, Image.cols, CUTOFF, Ord, Select, clStatus);
		break;
	case 2: printf("YOU HAVE SELECTED GAUSSIAN FILTER \n");
		program_smooth = Lowpass_Program(context, gaussian_kernel, device_list, clStatus);
		kernel_smooth = Lowpass_Kernel(program_smooth, kernel_smooth, "gaussian_kernel", clStatus);
		Kernel_Arg(kernel_smooth, Trans_RED_clmem, Trans_GREEN_clmem, Trans_BLUE_clmem, Image.rows, Image.cols, CUTOFF, Ord, Select, clStatus);
		break;
	case 3: printf("YOU HAVE SELECTED BUTTERWORTH FILTER \n");
		program_smooth = Lowpass_Program(context, butterworth_kernel, device_list, clStatus);
		kernel_smooth = Lowpass_Kernel(program_smooth, kernel_smooth, "butterworth_kernel", clStatus);
		Kernel_Arg(kernel_smooth, Trans_RED_clmem, Trans_GREEN_clmem, Trans_BLUE_clmem, Image.rows, Image.cols, CUTOFF, Ord, Select, clStatus);
		break;
	default:printf("YOU HAVE SELECTED WRONG FILTER \n");
		break;
	}

	// Start The Timer
	start = clock();

	// Extract [Red, Green, Blue] Channels from The Input Distorted Image
	split(Image, RGB_Image);

	// Copy Image Data to Host Buffers for Each Channels Separately.
	memcpy(RED_Frame, RGB_Image[0].data, Image.rows * Image.cols * sizeof(uchar));
	memcpy(GREEN_Frame, RGB_Image[1].data, Image.rows * Image.cols * sizeof(uchar));
	memcpy(BLUE_Frame, RGB_Image[2].data, Image.rows * Image.cols * sizeof(uchar));

	// Perform DFT Transforms [Horizontal DFT: Row wise DFT] + [Vertical DFT : Column wise DFT]
	Exec_Kernel(command_queue_smooth, kernel_HDFT, Image.rows*Image.cols, Image.rows, clStatus);
	Exec_Kernel(command_queue_smooth, kernel_VDFT, Image.rows*Image.cols, Image.rows, clStatus);
	
	//Execute the OpenCL Kernels for Smoothing of Each Channels Independetly.
	Exec_Kernel(command_queue_smooth, kernel_smooth, Image.rows*Image.cols, 1024, clStatus);

	// Perform DFT Transforms [Vertical IDFT : Column wise IDFT] + [Horizontal IDFT: Row wise IDFT]
	Exec_Kernel(command_queue_smooth, kernel_VIDFT, Image.rows*Image.cols, Image.rows, clStatus);
	Exec_Kernel(command_queue_smooth, kernel_HIDFT, Image.rows*Image.cols, Image.rows, clStatus);

	// Copy from Device Buffers to Host Buffers after Image Smoothing Operation.
	clStatus = clEnqueueReadBuffer(command_queue_smooth, RED_clmem, CL_TRUE, 0, Image.rows * Image.cols * sizeof(uchar), RED_Frame, 0, NULL, NULL);
	clStatus = clEnqueueReadBuffer(command_queue_smooth, GREEN_clmem, CL_TRUE, 0, Image.rows * Image.cols * sizeof(uchar), GREEN_Frame, 0, NULL, NULL);
	clStatus = clEnqueueReadBuffer(command_queue_smooth, BLUE_clmem, CL_TRUE, 0, Image.rows * Image.cols * sizeof(uchar), BLUE_Frame, 0, NULL, NULL);

	// Copy from Host Buffers to Image Variables for Each Channels Separately.
	memcpy(RGB_Image[0].data, RED_Frame, Image.rows * Image.cols * sizeof(uchar));
	memcpy(RGB_Image[1].data, GREEN_Frame, Image.rows * Image.cols * sizeof(uchar));
	memcpy(RGB_Image[2].data, BLUE_Frame, Image.rows * Image.cols * sizeof(uchar));

	// Merge All Three Channels to Construct Final Image
	merge(RGB_Image, 3, Image);

	//Stop the Timer
	end = clock();

	// Calculate Time Taken for Image Smoothing by a Low Pass Filter in millisecs.
	Time_Used = (float)(end - start) / CLOCKS_PER_SEC;
	printf("Time Taken for Image smoothing : %.3f", Time_Used);

	// Display Final Smoothed Image By Low Pass FIlter
	namedWindow("SMOOTHED IMAGE", WINDOW_AUTOSIZE);
	imshow("SMOOTHED IMAGE", Image);

	// Finally Release All OpenCL Allocated Objects and Buffers [Host & Device].
	releaseKernel(kernel_smooth, clStatus);
	releaseKernel(kernel_HDFT, clStatus);
	releaseKernel(kernel_HIDFT, clStatus);
	releaseKernel(kernel_VDFT, clStatus);
	releaseKernel(kernel_VIDFT, clStatus);
	clStatus = clReleaseProgram(program_smooth);
	clStatus = clReleaseProgram(program_HDFT);
	clStatus = clReleaseProgram(program_HIDFT);
	clStatus = clReleaseProgram(program_VDFT);
	clStatus = clReleaseProgram(program_VIDFT);
	clStatus = clReleaseMemObject(RED_clmem);
	clStatus = clReleaseMemObject(GREEN_clmem);
	clStatus = clReleaseMemObject(BLUE_clmem);
	clStatus = clReleaseMemObject(Trans_RED_clmem);
	clStatus = clReleaseMemObject(Trans_GREEN_clmem);
	clStatus = clReleaseMemObject(Trans_BLUE_clmem);
	clStatus = clReleaseMemObject(TR_temp);
	clStatus = clReleaseMemObject(TG_temp);
	clStatus = clReleaseMemObject(TB_temp);
	clStatus = clReleaseCommandQueue(command_queue_smooth);
	clStatus = clReleaseContext(context);
	free(RED_Frame);
	free(GREEN_Frame);
	free(BLUE_Frame);
	free(platforms);
	free(device_list);

	waitKey(0);
	return 0;
}

//++++++++++++++++++++++++++++++++++++++ END MAIN PROGRAM ++++++++++++++++++++++++++++++++++++++++++

