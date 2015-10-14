#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <CL/cl.h>

#define ARRAY_SIZE 4096
#define MAX_SOURCE_SIZE (0x100000)


// gcc -std=c99 vectors_cl.c -o vectors_cl -l OpenCL
// ./vectors_cl


int main(void) {
    const size_t ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

    // Generate the input array on the host.
    float h_a[ARRAY_SIZE];
    float h_b[ARRAY_SIZE];
    for (int i = 0; i < ARRAY_SIZE; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(2 * i);
    }

    float h_c[ARRAY_SIZE];

    FILE *fp;
    char *source_str;
    size_t source_size;

    fp = fopen("vectors_cl.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char *)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);

    // Get platform and device information
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1,
                         &device_id, &ret_num_devices);

    // Create an OpenCL context
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

    // Create memory buffers on the device for each vector
    cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                      ARRAY_BYTES, NULL, &ret);
    cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                      ARRAY_BYTES, NULL, &ret);
    cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                      ARRAY_BYTES, NULL, &ret);

    // Copy h_a and h_b to memory buffer
    ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0,
                               ARRAY_BYTES, h_a, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0,
                               ARRAY_BYTES, h_b, 0, NULL, NULL);

    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1,
        (const char **)&source_str, (const size_t *)&source_size, &ret);
    if (ret != 0) {
        printf("clCreateProgramWithSource returned non-zero status %d\n\n", ret);
        exit(1);
    }

    // Build the program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (ret != 0) {
        printf("clBuildProgram returned non-zero status %d: ", ret);

        if (ret == CL_INVALID_PROGRAM) {
            printf("invalid program\n");
        } else if (ret == CL_INVALID_VALUE) {
            printf("invalid value\n");
        } else if (ret == CL_INVALID_DEVICE) {
            printf("invalid device\n");
        } else if (ret == CL_INVALID_BINARY) {
            printf("invalid binary\n");
        } else if (ret == CL_INVALID_BUILD_OPTIONS) {
            printf("invalid build options\n");
        } else if (ret == CL_INVALID_OPERATION) {
            printf("invalid operation\n");
        } else if (ret == CL_COMPILER_NOT_AVAILABLE) {
            printf("compiler not available\n");
        } else if (ret == CL_BUILD_PROGRAM_FAILURE) {
            printf("build program failure\n");

            // Determine the size of the log
            size_t log_size;
            clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

            // Allocate memory for the log
            char *log = (char *) malloc(log_size);

            // Get the log
            clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

            // Print the log
            printf("%s\n", log);
        } else if (ret == CL_OUT_OF_HOST_MEMORY) {
            printf("out of host memory\n");
        }
        exit(1);
    }

    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "add", &ret);

    // Set the arguments of the kernel
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a_mem_obj);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&b_mem_obj);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&c_mem_obj);
    size_t array_size = ARRAY_SIZE;
    ret = clSetKernelArg(kernel, 3, sizeof(const size_t), (void *)&array_size);

    // Execute the OpenCL kernel on the list
    size_t global_item_size = ARRAY_SIZE; // Process the entire lists
    size_t local_item_size = 1; // Divide work items into groups of 64
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
            &global_item_size, &local_item_size, 0, NULL, NULL);

    // Read the memory buffer C on the device to the local variable C
    ret = clEnqueueReadBuffer(command_queue, c_mem_obj, CL_TRUE, 0,
                              ARRAY_BYTES, h_c, 0, NULL, NULL);

    // Print out the resulting array.
    for (int i = 0; i < 8; i++) {
        printf("%d + %d = %d", (int)h_a[i], (int)h_b[i], (int)h_c[i]);
        printf(((i % 4) != 3) ? "\t" : "\n");
    }

    printf("...\n");

    for (int i = ARRAY_SIZE - 8; i < ARRAY_SIZE; i++) {
        printf("%d + %d = %d",
               (int)h_a[i], (int)h_b[i], (int)h_c[i]);
        printf(((i % 4) != 3) ? "\t" : "\n");
    }

    // Clean up
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(a_mem_obj);
    ret = clReleaseMemObject(b_mem_obj);
    ret = clReleaseMemObject(c_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);

    return 0;
}
