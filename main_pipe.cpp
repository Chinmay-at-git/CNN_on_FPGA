#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "CL/opencl.h"
#include "AOCL_Utils.h"
#include <string.h>
#define AOCL_ALIGNMENT 64
using namespace aocl_utils;

// OpenCL runtime configuration
cl_platform_id platform = NULL;
unsigned num_devices = 0;
cl_device_id *device; 
cl_context context = NULL;
cl_command_queue queue,queue1; 
cl_program program = NULL;
cl_kernel kernel; 
cl_program program1 = NULL;
cl_kernel kernel1; 

cl_mem input_a_buf; 
cl_mem input_b_buf; 
cl_mem output_buf; 
cl_mem softmax_buf;
cl_mem Weights_buf;
cl_mem Wsoft_buf;
// Problem data.
int sigm=10,softmax_no=10;
int N,gRows,gCols,filtsize,HFS; // problem size
float *input_a; 
float input_b[]={1,1,1,1,-8,1,1,1,1};
float *output;
float *ref_output;
float *Weights_local, *Wsoft_local, *softmax_local;
// Function prototypes
float rand_float();
bool init_opencl();
void init_problem(char *infile,int,int,int);
void run(char *outFile);
void cleanup();

// Entry point.
int main(int argc, char **argv) {
  if(argc < 6){
	printf("Usage is: ./convolution INFILE OUTFILE ROWS COLS CHANNELS\n");
	return -1;
  }

  // Initialize the problem data.
  init_problem(argv[1],atoi(argv[3]),atoi(argv[4]),atoi(argv[5]));
	
  // Initialize OpenCL.
  if(!init_opencl()) {
    return -1;
  }

  // Run the kernel.
  run(argv[2]);

  // Free the resources allocated
  cleanup();

  return 0;
}

/////// HELPER FUNCTIONS ///////

// Initializes the OpenCL objects.
bool init_opencl() {
  cl_int status;

  printf("Initializing OpenCL\n");

  if(!setCwdToExeDir()) {
    return false;
  }

  // Get the OpenCL platform.
  platform = findPlatform("Altera");
  if(platform == NULL) {
    printf("ERROR: Unable to find Altera OpenCL platform.\n");
    return false;
  }

  // Query the available OpenCL device.
  device = getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices);
  printf("Platform: %s\n", getPlatformName(platform).c_str());
  printf("Using %d device(s)\n", num_devices);
  printf("  %s\n", getDeviceName(*device).c_str());

  // Create the context.
  context = clCreateContext(NULL, num_devices, device, NULL, NULL, &status);
  checkError(status, "Failed to create context");

  // Create the program for all device. Use the first device as the
  // representative device (assuming all device are of the same type).
  std::string binary_file = getBoardBinaryFile("cnnpipe", *device);
  printf("Using AOCX: %s\n", binary_file.c_str());
  program = createProgramFromBinary(context, binary_file.c_str(), device, num_devices);

  // Build the program that was just created.
  status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
  checkError(status, "Failed to build program");

  // Command queue.
  queue = clCreateCommandQueue(context, *device, CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status, "Failed to create command queue");

  // Command queue.
  queue1 = clCreateCommandQueue(context, *device, CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status, "Failed to create command queue");
  // Kernel.
  const char *kernel_name = "conv_kernel";
  kernel = clCreateKernel(program, kernel_name, &status);
  checkError(status, "Failed to create kernel");
//const char *kernel_name1 = "sigm_kernel1";
//  kernel1 = clCreateKernel(program, kernel_name1, &status);
//  checkError(status, "Failed to create kernel1");	
  // Input buffers.
  input_a_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, 
        N * sizeof(float), NULL, &status);

  checkError(status, "Failed to create buffer for input A");
Weights_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, 
       sigm* N * sizeof(float), NULL, &status);
  checkError(status, "Failed to create buffer for Weights");
Wsoft_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, 
       sigm* softmax_no * sizeof(float), NULL, &status);

  checkError(status, "Failed to create buffer for Wsoft");
input_b_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, 
        filtsize * sizeof(float), NULL, &status);
  checkError(status, "Failed to create buffer for inputb");
//Read-write buffers
//sigm_buf=clCreateBuffer(context, CL_MEM_READ_WRITE, 
//        filtsize * sizeof(float), NULL, &status);
//checkError(status, "Failed to create buffer for sigm");
  // Output buffer.
  softmax_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
        softmax_no * sizeof(float), NULL, &status);
  checkError(status, "Failed to create buffer for output");

  return true;
}

//Initialize data for the problem.
void init_problem(char* infile,int Rows,int Cols,int channels) {
	//Open File
	FILE *input = fopen(infile, "rb");
	
	if(input == NULL){
		printf("Failed to open input file\n");
		exit(0);	
	}

	//Read Size
	N=Rows*Cols*channels;
	gCols=Cols;
	gRows=Rows;
	filtsize=9;
	HFS=1;	
	//Allocate Arrays
	// R-G-B all floats Float32
	

#include <stdlib.h>
	//void *ptr = NULL;
	//posix_memalign (&ptr, AOCL_ALIGNMENT, size);


	//input_a = (float*)malloc(sizeof(float) * N);
		//output = (float*)malloc(sizeof(float) *  N);
	posix_memalign((void**)&input_a, AOCL_ALIGNMENT, sizeof(float) * N);
	//Weights_local=(float*)malloc(sizeof(float) * N * sigm);
	posix_memalign((void**)&Weights_local, AOCL_ALIGNMENT, sizeof(float) * N *sigm);
	//Wsoft_local=(float*)malloc(sizeof(float) * sigm * softmax_no);
	posix_memalign((void**)&Wsoft_local, AOCL_ALIGNMENT, sizeof(float) * sigm*softmax_no);
	//softmax_local=(float*)malloc(sizeof(float) * softmax_no);
	posix_memalign((void**)&softmax_local, AOCL_ALIGNMENT, sizeof(float) * softmax_no);	
	if(input_a == NULL || !Weights_local || !Wsoft_local || !softmax_local){
		printf("Failed to allocate host memory\n");
		exit(0);
	}
	//Read data
	fread(input_a, sizeof(float), N, input);
	
	printf("Convolving %d elements\n", N);
	fclose(input);
}

void run(char* outFile) {
  cl_int status;

  const double start_time = getCurrentTimestamp();

  // Launch the problem for each device.
  cl_event kernel_event;
  cl_event finish_event;


  // Transfer inputs to each device. Each of the host buffers supplied to
  // clEnqueueWriteBuffer here is already aligned to ensure that DMA is used
  // for the host-to-device transfer.
  cl_event write_event1,write_event2,write_event3,write_event4;
  status = clEnqueueWriteBuffer(queue, input_a_buf, CL_FALSE,
       0, N * sizeof(float), input_a, 0, NULL, &write_event1);
  checkError(status, "Failed to transfer input A");


  status = clEnqueueWriteBuffer(queue, input_b_buf, CL_FALSE,
       0, filtsize * sizeof(float), input_b, 0, NULL, &write_event2);
  checkError(status, "Failed to transfer inb");
  status = clEnqueueWriteBuffer(queue, Weights_buf, CL_FALSE,
       0, filtsize * sizeof(float), Weights_local, 0, NULL, &write_event3);
  checkError(status, "Failed to transfer weights");
  status = clEnqueueWriteBuffer(queue, Wsoft_buf, CL_FALSE,
       0, filtsize * sizeof(float), Wsoft_local, 0, NULL, &write_event4);
  checkError(status, "Failed to transfer wsofts");
 cl_event writes[]= {write_event1,write_event2,write_event3,write_event4};
  clWaitForEvents(4, writes);
  // Set kernel arguments.
  unsigned argi = 0;

  status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &input_a_buf);
  checkError(status, "Failed to set argument %d", argi - 1);

  status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &Weights_buf);
  checkError(status, "Failed to set argument %d", argi - 1);

  status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &Wsoft_buf);
  checkError(status, "Failed to set argument %d", argi - 1);

  status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &softmax_buf);
  checkError(status, "Failed to set argument %d", argi - 1);
status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &input_b_buf);
  checkError(status, "Failed to set argument %d", argi - 1);
		

  status = clSetKernelArg(kernel, argi++, sizeof(int), &gRows);
  status = clSetKernelArg(kernel, argi++, sizeof(int), &gCols);
  checkError(status, "Failed to set argument %d", argi - 1);


  status = clSetKernelArg(kernel, argi++, sizeof(int), &sigm);
  checkError(status, "Failed to set argument %d", argi - 1);

  status = clSetKernelArg(kernel, argi++, sizeof(int), &softmax_no);
  checkError(status, "Failed to set argument %d", argi - 1);
//set Kernel-1 arguments
/*argi=0;
status = clSetKernelArg(kernel1, argi++, sizeof(cl_mem), &Weights_buf);
  checkError(status, "Failed to set argument %d", argi - 1);

status = clSetKernelArg(kernel1, argi++, sizeof(cl_mem), &output_buf);
  checkError(status, "Failed to set argument %d", argi - 1);
status = clSetKernelArg(kernel1, argi++, sizeof(cl_mem), &sigm_buf);
  checkError(status, "Failed to set argument %d", argi - 1);

status = clSetKernelArg(kernel1, argi++, sizeof(int), &gRows);
 checkError(status, "Failed to set argument %d", argi - 1);
status = clSetKernelArg(kernel1, argi++, sizeof(int), &gCols);
  checkError(status, "Failed to set argument %d", argi - 1);
status = clSetKernelArg(kernel1, argi++, sizeof(int), &sigm);
  checkError(status, "Failed to set argument %d", argi - 1);
*/


  // Enqueue kernel.
  // Use a global work size corresponding to the number of elements to add
  // for this device.
  // 
  // We don't specify a local work size and let the runtime choose
  // (it'll choose to use one work-group with the same size as the global
  // work-size).
  //
  // Events are used to ensure that the kernel is not launched until
  // the writes to the input buffers have completed.
  const size_t global_work_size[] ={gRows,gCols} ;
const size_t onework[]={1};
// clEnque parameters: Command_queue, kernel, work_dim, global_work_offset, global_work_size,local_work_size, event_wait_list,num_eventsinwaitlist,event
  printf("Launching for device %d (%d %delements)\n", 0, global_work_size[0],global_work_size[1]);
  status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL,
        onework, NULL, 4, writes, &kernel_event);
printf("\nLaunched first kernel!!!!! ");
//status = clEnqueueNDRangeKernel(queue1, kernel1, 2, NULL,
  //      global_work_size, NULL, 0, NULL, &kernel_event);
  //checkError(status, "Failed to launch kernel");
//printf("\nLaunched second one");

  clWaitForEvents(num_devices, &kernel_event);
  
  // Read the result. This the final operation.
  status = clEnqueueReadBuffer(queue, softmax_buf, CL_FALSE,
        0, softmax_no * sizeof(float), softmax_local, 1, &kernel_event, &finish_event);

  // Release local events.
  clReleaseEvent(write_event1);
  

  // Wait for all devices to finish.
  clWaitForEvents(num_devices, &finish_event);

  const double end_time = getCurrentTimestamp();

  // Wall-clock time taken.
  printf("\nTime: %0.3f ms\n", (end_time - start_time) * 1e3);

  // Get kernel times using the OpenCL event profiling API.
  cl_ulong time_ns = getStartEndTime(kernel_event);
  printf("Kernel time (device 0): %0.3f ms\n", double(time_ns) * 1e-6);
  
  // Release all events.
  clReleaseEvent(kernel_event);
  clReleaseEvent(finish_event);
  
  printf("Writing Results to File %s\n", outFile);

  FILE *outputFile = fopen(outFile, "wb");
  if(outputFile == NULL){
	printf("Failed to open output file");
  }else{
	fwrite(output,sizeof(float),N,outputFile);
  	
  }  

}

// Free the resources allocated during initialization
void cleanup() {
  if(kernel) {
    clReleaseKernel(kernel);
  }
if(kernel1)
	clReleaseKernel(kernel1);
  if(queue) {
      clReleaseCommandQueue(queue);
  }
 if(queue1)
	clReleaseCommandQueue(queue1);
  if(input_a_buf) {
      clReleaseMemObject(input_a_buf);
  }
  if(input_b_buf) {
      clReleaseMemObject(input_b_buf);
  }
  if(output_buf) {
      clReleaseMemObject(output_buf);
  }
  

  if(program) {
    clReleaseProgram(program);
  }
  if(context) {
    clReleaseContext(context);
  }
}


