#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <unistd.h>
#include <math.h>
#include <algorithm>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

using namespace std;

const char *getErrorString(cl_int error)
{
switch(error){
    // run-time and JIT compiler errors
    case 0: return "CL_SUCCESS";
    case -1: return "CL_DEVICE_NOT_FOUND";
    case -2: return "CL_DEVICE_NOT_AVAILABLE";
    case -3: return "CL_COMPILER_NOT_AVAILABLE";
    case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case -5: return "CL_OUT_OF_RESOURCES";
    case -6: return "CL_OUT_OF_HOST_MEMORY";
    case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case -8: return "CL_MEM_COPY_OVERLAP";
    case -9: return "CL_IMAGE_FORMAT_MISMATCH";
    case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case -11: return "CL_BUILD_PROGRAM_FAILURE";
    case -12: return "CL_MAP_FAILURE";
    case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case -15: return "CL_COMPILE_PROGRAM_FAILURE";
    case -16: return "CL_LINKER_NOT_AVAILABLE";
    case -17: return "CL_LINK_PROGRAM_FAILURE";
    case -18: return "CL_DEVICE_PARTITION_FAILED";
    case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

    // compile-time errors
    case -30: return "CL_INVALID_VALUE";
    case -31: return "CL_INVALID_DEVICE_TYPE";
    case -32: return "CL_INVALID_PLATFORM";
    case -33: return "CL_INVALID_DEVICE";
    case -34: return "CL_INVALID_CONTEXT";
    case -35: return "CL_INVALID_QUEUE_PROPERTIES";
    case -36: return "CL_INVALID_COMMAND_QUEUE";
    case -37: return "CL_INVALID_HOST_PTR";
    case -38: return "CL_INVALID_MEM_OBJECT";
    case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case -40: return "CL_INVALID_IMAGE_SIZE";
    case -41: return "CL_INVALID_SAMPLER";
    case -42: return "CL_INVALID_BINARY";
    case -43: return "CL_INVALID_BUILD_OPTIONS";
    case -44: return "CL_INVALID_PROGRAM";
    case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case -46: return "CL_INVALID_KERNEL_NAME";
    case -47: return "CL_INVALID_KERNEL_DEFINITION";
    case -48: return "CL_INVALID_KERNEL";
    case -49: return "CL_INVALID_ARG_INDEX";
    case -50: return "CL_INVALID_ARG_VALUE";
    case -51: return "CL_INVALID_ARG_SIZE";
    case -52: return "CL_INVALID_KERNEL_ARGS";
    case -53: return "CL_INVALID_WORK_DIMENSION";
    case -54: return "CL_INVALID_WORK_GROUP_SIZE";
    case -55: return "CL_INVALID_WORK_ITEM_SIZE";
    case -56: return "CL_INVALID_GLOBAL_OFFSET";
    case -57: return "CL_INVALID_EVENT_WAIT_LIST";
    case -58: return "CL_INVALID_EVENT";
    case -59: return "CL_INVALID_OPERATION";
    case -60: return "CL_INVALID_GL_OBJECT";
    case -61: return "CL_INVALID_BUFFER_SIZE";
    case -62: return "CL_INVALID_MIP_LEVEL";
    case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
    case -64: return "CL_INVALID_PROPERTY";
    case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
    case -66: return "CL_INVALID_COMPILER_OPTIONS";
    case -67: return "CL_INVALID_LINKER_OPTIONS";
    case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

    // extension errors
    case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
    case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
    case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
    case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
    case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
    case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
    default: return "Unknown OpenCL error";
    }
}

#define CHECK_CL_ERROR(e) \
if (e != CL_SUCCESS) \
{ \
	cout << "cl ERROR " << getErrorString(e) << " : line " << __LINE__ << endl; \
}

#define SIZE_OF_MATRIX 1000
#define SIZE_OF_BLOCK 16

#define M SIZE_OF_MATRIX

unsigned int m = SIZE_OF_MATRIX;

int source_lines = 67;
const char* multiply_matrices_source[] = {

"#define SIZE_OF_BLOCK 16\n",
"int idx(int i, int j, int lda)\n",
"{\n",
	"return ((j) + ((i)*(lda)));\n",
"}\n",
"__kernel void null_kernel()\n",
"{}\n",
"__kernel void multiply_matrices(__global const float *d_a, __global const float *d_b, __global float *d_c, int lda)\n",
"{\n",
	"unsigned int row = get_global_id(1);\n",
	"unsigned int col = get_global_id(0);\n",
	"unsigned int id  = idx(row,col,lda);\n",
	"float ctemp;\n",
	"if (row < lda && col < lda)\n",
	"{\n",
		"ctemp = 0;\n",
		"for (unsigned int j=0; j<lda; j++)\n",
		"{\n",
			"ctemp = ctemp + d_a[idx(row,j,lda)] * d_b[idx(j,col,lda)];\n",
		"}\n",
		"d_c[id] = ctemp;\n",
	"}\n",
"}\n",
"__kernel void multiply_matrices_shared_blocks(__global float *d_a, __global float *d_b, __global float *d_c, __local float* a, __local float* b, int lda)\n",
"{\n",
" int bs = SIZE_OF_BLOCK;",
"\n",
"	unsigned int row = get_global_id(1);\n",
"	unsigned int col = get_global_id(0);\n",
"	unsigned int id  = idx(row,col,lda);\n",
"	\n",
"\n",
"	//temp element of d_c\n",
"	float c = 0;\n",
"	\n",
"	//top-level row,col of block\n",
"	int block_row = get_group_id(1) * bs;\n",
"	int block_col = get_group_id(0) * bs;\n",
"\n",
"	//id inside each block\n",
"	int sub_row = get_local_id(1);\n",
"	int sub_col = get_local_id(0);\n",
"	\n",
"	//for each block	\n",
"	for (int k = 0; k < (lda / bs); k++)\n",
"	{\n",
"\n",
"		a[idx(sub_row, sub_col, lda)] = a[idx(row, col, lda)];\n",
"		b[idx(sub_row, sub_col, lda)] = b[idx(row, col, lda)];\n",
"		\n",
"		//wait for all threads to complete copy to shared memory.	\n",
"		barrier(CLK_LOCAL_MEM_FENCE);\n",
"\n",
"		//multiply each submatrix\n",
"		/*for (int j=0; j < bs; j++)\n",
"		{\n",
"			c = c + a[idx(sub_row, j, lda)] * b[idx(j, sub_col, lda)];\n",
"		}*/\n",
"	\n",
"		// move results to device memory.\n",
"		d_c[id] = c;\n",
"	\n",
"		// wait for multiplication to finish before moving onto the next submatrix.\n",
"		barrier(CLK_LOCAL_MEM_FENCE);\n",
"		\n",
"	}\n",
"}\n",
};

int main(int argc, char**argv)
{
	unsigned int number_of_threads = min(SIZE_OF_MATRIX, SIZE_OF_BLOCK);
	unsigned int number_of_blocks, block_mult;
	if (SIZE_OF_MATRIX > SIZE_OF_BLOCK)
		block_mult = ceil(SIZE_OF_MATRIX / ((float) SIZE_OF_BLOCK));
	else
		 block_mult = 1;


	number_of_blocks = SIZE_OF_BLOCK * block_mult;

	unsigned int matsize = SIZE_OF_MATRIX*SIZE_OF_MATRIX*sizeof(float);
	unsigned int submatsize = SIZE_OF_BLOCK*SIZE_OF_BLOCK*sizeof(float);

	//std::cout << "blocks: " << number_of_blocks << " threads: " <<
	//number_of_threads << std::endl;

	std::cout.flush();

	float* a = (float*)malloc(matsize);
	float* b = (float*)malloc(matsize);
	float* c = (float*)malloc(matsize);

	//initalize matrices
	for (int i=0; i<m; i++) {
		for (int j=0; j<m; j++) {
			//a[i*m+j] = i;
			//b[i*m+j] = i;
			a[i*m+j] = i-j*2 + i-j+1 + 1;
			b[i*m+j] = i-j*2 + i-j+1 + 1;
			c[i*m+j] = 0;
			//std::cout << a[i*m+j] << ", ";
		}
		//std::cout << std::endl;
	}

	cl_int ci;
	cl_platform_id cpPlatform[1];
    cl_uint num_platforms;

	ci = clGetPlatformIDs(1, cpPlatform, &num_platforms);
	CHECK_CL_ERROR(ci);
	cout << num_platforms << " platforms found." << endl;
    if (num_platforms == 0) exit(-1);

    //for (auto p =0 ; p < num_platforms ; p++) {
    for (auto p =0 ; p < 1 ; p++) {
	cl_uint nDevices, count;
	cl_device_id *cdDevices = NULL;
	ci = clGetDeviceIDs(cpPlatform[p], CL_DEVICE_TYPE_GPU, 0, NULL, &count);

	cdDevices = (cl_device_id *)malloc(count * sizeof(cl_device_id));
	ci = clGetDeviceIDs(cpPlatform[p], CL_DEVICE_TYPE_GPU, count, cdDevices, NULL);
	//ci = clGetDeviceIDs(cpPlatform[p], CL_DEVICE_TYPE_CPU, 1, &cdDevice, NULL);
	CHECK_CL_ERROR(ci);

	cout << count << " devices found." << endl;
    if (count == 0) exit(-1);

	string device_list("");
	int number_of_iterations = 1;

	int opt = getopt(argc, argv, "d:i:");
	while(opt != -1) {
		stringstream str;
		switch(opt) {
			case 'd':
				device_list = string(optarg);
				break;
			case 'i':
				str << optarg;
				str >> number_of_iterations;
				break;
			case '?':
				if (optopt == 'd')
					cerr << "Error, option -d requires argument: comma delimted list of devices to run on." << endl;
				else if (optopt == 'i')
					cerr << "Error, option -i requires argument: number of iterations to run." << endl;
				else
					cerr << "Error, unknow option. Usage:\nmatmult [-d <device id>,...] [-i <number of iterations]" << endl;
				return 1;
			default:
				break;
		}
	  opt = getopt(argc, argv, "d:i:");
	}
	cl_device_id* devices = (cl_device_id*) malloc(count * sizeof(cl_device_id));
	nDevices = 0;
	//default: use all the devices
	if (device_list.compare("") == 0)
	{
		for (int d=0;d<count;d++)
		{
			devices[d] = cdDevices[d];
		}
		nDevices = count;
	}
	else
	{
		for (int d=0;d<count;d++)
		{
			stringstream str;
			str << d;
			char c = 0;
			if (str >> c) {
				if (device_list.find(c) != string::npos) {
					devices[nDevices++] = cdDevices[d];
				}
			}
		}
	}
	//cout << "finnished mapping devices." << endl;

	//cl_context GPUContext = clCreateContextFromType(0, CL_DEVICE_TYPE_GPU, NULL, NULL, &ci);
	cl_context GPUContext = clCreateContext(0, nDevices, devices, NULL, NULL, &ci);
	CHECK_CL_ERROR(ci);

	cl_command_queue cqCommandQueue[nDevices];

	for (int d=0;d<nDevices;d++)
	{
		char name[256];
		clGetDeviceInfo(devices[d], CL_DEVICE_NAME, sizeof(name), &name, NULL);
		cout << "Using device name: " << name << endl;

		cqCommandQueue[d] = clCreateCommandQueue(GPUContext, devices[0], CL_QUEUE_PROFILING_ENABLE, &ci);
		CHECK_CL_ERROR(ci);

	}



	cl_program OpenCLProgram = clCreateProgramWithSource(GPUContext, source_lines, multiply_matrices_source, NULL, &ci);
	CHECK_CL_ERROR(ci);

	ci = clBuildProgram(OpenCLProgram, 0, NULL, NULL, NULL, NULL);
	CHECK_CL_ERROR(ci);

	char log[60000];
	ci = clGetProgramBuildInfo(OpenCLProgram, devices[0], CL_PROGRAM_BUILD_LOG,
	60000, log, NULL);

	CHECK_CL_ERROR(ci);

	//printf("build log: %s\n", log);
	//cout << log << endl;

	size_t thread_size[] = {number_of_threads, number_of_threads};
	size_t block_size[] = {number_of_blocks, number_of_blocks};
  /*
	cl_mem sub_a = clCreateBuffer(GPUContext, CL_MEM_ALLOC_HOST_PTR, submatsize,
	NULL, NULL);
	cl_mem sub_b = clCreateBuffer(GPUContext, CL_MEM_ALLOC_HOST_PTR, submatsize,
	NULL, NULL);

	cl_kernel OpenCL_multiply_matrices_shared_blocks = clCreateKernel(OpenCLProgram,
	"multiply_matrices_shared_blocks", &ci);

	CHECK_CL_ERROR(ci);

	ci = clSetKernelArg(OpenCL_multiply_matrices_shared_blocks, 0, sizeof(cl_mem), (void *) &d_a);
	CHECK_CL_ERROR(ci);
	ci = clSetKernelArg(OpenCL_multiply_matrices_shared_blocks, 1, sizeof(cl_mem), (void *) &d_b);
	CHECK_CL_ERROR(ci);
	ci = clSetKernelArg(OpenCL_multiply_matrices_shared_blocks, 2, sizeof(cl_mem), (void *) &d_c);
	CHECK_CL_ERROR(ci);
	ci = clSetKernelArg(OpenCL_multiply_matrices_shared_blocks, 3, sizeof(float) * SIZE_OF_BLOCK * SIZE_OF_BLOCK, 0);
	CHECK_CL_ERROR(ci);
	ci = clSetKernelArg(OpenCL_multiply_matrices_shared_blocks, 4, sizeof(float) * SIZE_OF_BLOCK * SIZE_OF_BLOCK, 0);
	CHECK_CL_ERROR(ci);
	ci = clSetKernelArg(OpenCL_multiply_matrices_shared_blocks, 5, sizeof(int), (void *) &m);
	CHECK_CL_ERROR(ci);


	cl_event event, shared_event;
	ci = clEnqueueNDRangeKernel(cqCommandQueue,
	OpenCL_multiply_matrices_shared_blocks, 2, NULL, block_size, thread_size, 0, NULL, &shared_event);
	//cl_kernel OpenCL_null_kernel = clCreateKernel(OpenCLProgram,
	//"null_kernel", &ci);
	//CHECK_CL_ERROR(ci);
	//ci = clEnqueueNDRangeKernel(cqCommandQueue, OpenCL_null_kernel, 2, NULL, block_size, thread_size, 0, NULL, &shared_event);
	CHECK_CL_ERROR(ci);
	*/
	cl_kernel OpenCL_multiply_matrices = clCreateKernel(OpenCLProgram, "multiply_matrices", &ci);
	CHECK_CL_ERROR(ci);

	cl_event event, event_mem[2], event_read;
	cl_mem d_a, d_b, d_c;

	for (int i=0; i<number_of_iterations*nDevices; i++)
	{
		cl_command_queue cCQ = cqCommandQueue[i%nDevices];

		d_a = clCreateBuffer(GPUContext, CL_MEM_READ_ONLY |
		CL_MEM_COPY_HOST_PTR, matsize, a, &ci);
		CHECK_CL_ERROR(ci);
		d_b = clCreateBuffer(GPUContext, CL_MEM_READ_ONLY |
		CL_MEM_COPY_HOST_PTR, matsize, b, &ci);
		CHECK_CL_ERROR(ci);

		d_c = clCreateBuffer(GPUContext, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, matsize, c, &ci);
		CHECK_CL_ERROR(ci);

		clSetKernelArg(OpenCL_multiply_matrices, 0, sizeof(cl_mem), (void *) &d_a);
		clSetKernelArg(OpenCL_multiply_matrices, 1, sizeof(cl_mem), (void *) &d_b);
		clSetKernelArg(OpenCL_multiply_matrices, 2, sizeof(cl_mem), (void *) &d_c);
		clSetKernelArg(OpenCL_multiply_matrices, 3, sizeof(int), (void *) &m);

		event_mem[0] = clCreateUserEvent(GPUContext, &ci);
		CHECK_CL_ERROR(ci);
		clEnqueueWriteBuffer(cCQ, d_a, CL_TRUE, 0, matsize, a, 0, NULL, &(event_mem[0]));
		event_mem[1] = clCreateUserEvent(GPUContext, &ci);
		CHECK_CL_ERROR(ci);
		clEnqueueWriteBuffer(cCQ, d_b, CL_TRUE, 0, matsize, b, 0, NULL, &(event_mem[1]));
		//clWaitForEvents(2, event_mem);

		event = clCreateUserEvent(GPUContext, &ci);
		CHECK_CL_ERROR(ci);

		ci = clEnqueueNDRangeKernel(cCQ, OpenCL_multiply_matrices, 2, NULL,
		block_size, thread_size, 2, event_mem, &event);
		CHECK_CL_ERROR(ci);

		//clWaitForEvents(1, &shared_event);
		clWaitForEvents(1, &event);
		CHECK_CL_ERROR(ci);

		event_read = clCreateUserEvent(GPUContext, &ci);
		//ci = clEnqueueReadBuffer(cCQ, d_c, CL_TRUE, 0, matsize, c, 0, NULL, &event_read);
		ci = clEnqueueReadBuffer(cCQ, d_c, CL_TRUE, 0, matsize, c, 0, NULL, NULL);
		CHECK_CL_ERROR(ci);
		//clWaitForEvents(1, &event_read);
		//clFinish(cCQ);

	}

	cout << "Finished " << number_of_iterations << " iterations on " << nDevices << " devices." << endl;
	/*
	std::cout << " results: " << std::endl;
	for (int i=0; i<m; i++) {
		for (int j=0; j<m; j++) {
			std::cout << c[i*m+j] << ", ";
		}
		std::cout << std::endl;
	}
	*/

	free(a);
	free(b);
	free(c);

	clReleaseKernel(OpenCL_multiply_matrices);
	clReleaseProgram(OpenCLProgram);
	for (int d=0;d<nDevices;d++)
	{
		clFinish(cqCommandQueue[d]);
		clReleaseCommandQueue(cqCommandQueue[d]);
	}
	clReleaseContext(GPUContext);
	clReleaseMemObject(d_a);
	clReleaseMemObject(d_b);
	clReleaseMemObject(d_c);
    }

}
