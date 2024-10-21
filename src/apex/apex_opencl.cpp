/*******************************************************************************
 * Copyright (c) 2008-2020 The Khronos Group Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#ifndef CL_API_PREFIX__VERSION_1_1_DEPRECATED
#define CL_API_PREFIX__VERSION_1_1_DEPRECATED
#endif
#ifndef CL_API_SUFFIX__VERSION_1_1_DEPRECATED
#define CL_API_SUFFIX__VERSION_1_1_DEPRECATED
#endif
#ifndef CL_API_PREFIX__VERSION_1_2_DEPRECATED
#define CL_API_PREFIX__VERSION_1_2_DEPRECATED
#endif
#ifndef CL_API_SUFFIX__VERSION_1_2_DEPRECATED
#define CL_API_SUFFIX__VERSION_1_2_DEPRECATED
#endif
#else
#include <CL/cl.h>
#endif

#include <type_traits>
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include "apex_api.hpp"

namespace apex {
namespace opencl {

void* get_library(void) {
    /* If, for some reason, we haven't preloaded the libXXX.so
     * library, Go looking for the library, and then the symbol.
     * This assumes that the LD_LIBRARY_PATH will include the
     * path to the library. */

    /* Check for the environment variable */
#ifdef __APPLE__
    char const * libname = "/System/Library/Frameworks/OpenCL.framework/OpenCL";
#else
    char const * libname = "libOpenCL.so";
#endif /* __APPLE__ */

    /* Check to see if we've already loaded it */
    void* handle = dlopen(libname, RTLD_NOLOAD | RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
        /* Library hasn't been loaded, so load it */
        handle = dlopen(libname, RTLD_NOW | RTLD_LOCAL);
    }
    if (handle == nullptr) {
        fprintf(stderr, "Unable to load library %s\n%s\n", libname, dlerror());
        return nullptr;
    }
    return handle;
}

void* get_symbol(const char * symbol) {
    // Do we have the symbol already?
    //void* func = dlsym(RTLD_DEFAULT, symbol);
    //if (func != nullptr) {
        //return func;
    //}
    // look up the library
    static void* library_handle = get_library();
    // Do we have a library to search?
    if (library_handle == nullptr) {
        return nullptr;
    }
    void* func = dlsym(library_handle, symbol);
    if (func == nullptr) {
        fprintf(stderr, "Unable to load symbol %s\n%s\n", symbol, dlerror());
        return nullptr;
    }
    return func;
}

template<typename T>
T* getsym(const char * name) {
    T* p = reinterpret_cast<T*>(get_symbol(name));
    return p;
}

} // namespace opencl
} // namespace apex

#define GET_SYMBOL(name) static decltype(name)* function_ptr = \
    apex::opencl::getsym<decltype(name)>(#name);

#define GET_SYMBOL_TIMER(name) static decltype(name)* function_ptr = \
    apex::opencl::getsym<decltype(name)>(#name); \
    apex::scoped_timer timer((uint64_t)function_ptr);

extern "C" {

/* Platform API */
extern CL_API_ENTRY cl_int CL_API_CALL
clGetPlatformIDs(cl_uint          num_entries,
                 cl_platform_id * platforms,
                 cl_uint *        num_platforms) CL_API_SUFFIX__VERSION_1_0 {
    GET_SYMBOL_TIMER(clGetPlatformIDs);
    return function_ptr(num_entries, platforms, num_platforms);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clGetPlatformInfo(cl_platform_id   platform,
                  cl_platform_info param_name,
                  size_t           param_value_size,
                  void *           param_value,
                  size_t *         param_value_size_ret) CL_API_SUFFIX__VERSION_1_0 {
    GET_SYMBOL_TIMER(clGetPlatformInfo);
    return function_ptr(platform, param_name, param_value_size, param_value, param_value_size_ret);
}


/* Device APIs */
extern CL_API_ENTRY cl_int CL_API_CALL
clGetDeviceIDs(cl_platform_id   platform,
               cl_device_type   device_type,
               cl_uint          num_entries,
               cl_device_id *   devices,
               cl_uint *        num_devices) CL_API_SUFFIX__VERSION_1_0 {
    GET_SYMBOL_TIMER(clGetDeviceIDs);
    return function_ptr(platform, device_type, num_entries, devices, num_devices);
}


extern CL_API_ENTRY cl_int CL_API_CALL
clGetDeviceInfo(cl_device_id    device,
                cl_device_info  param_name,
                size_t          param_value_size,
                void *          param_value,
                size_t *        param_value_size_ret) CL_API_SUFFIX__VERSION_1_0 {
    GET_SYMBOL_TIMER(clGetDeviceInfo);
    return function_ptr(device, param_name, param_value_size, param_value, param_value_size_ret);
}


#ifdef CL_VERSION_1_2

extern CL_API_ENTRY cl_int CL_API_CALL
clCreateSubDevices(cl_device_id                         in_device,
                   const cl_device_partition_property * properties,
                   cl_uint                              num_devices,
                   cl_device_id *                       out_devices,
                   cl_uint *                            num_devices_ret) CL_API_SUFFIX__VERSION_1_2 {
    GET_SYMBOL_TIMER(clCreateSubDevices);
    return function_ptr(in_device, properties, num_devices, out_devices, num_devices_ret);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clRetainDevice(cl_device_id device) CL_API_SUFFIX__VERSION_1_2 {
    GET_SYMBOL_TIMER(clRetainDevice);
    return function_ptr(device);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clReleaseDevice(cl_device_id device) CL_API_SUFFIX__VERSION_1_2 {
    GET_SYMBOL_TIMER(clReleaseDevice);
    return function_ptr(device);
}

#endif

#ifdef CL_VERSION_2_1

extern CL_API_ENTRY cl_int CL_API_CALL
clSetDefaultDeviceCommandQueue(cl_context           context,
                               cl_device_id         device,
                               cl_command_queue     command_queue) CL_API_SUFFIX__VERSION_2_1 {
    GET_SYMBOL_TIMER(clSetDefaultDeviceCommandQueue);
    return function_ptr(context, device, command_queue);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clGetDeviceAndHostTimer(cl_device_id    device,
                        cl_ulong*       device_timestamp,
                        cl_ulong*       host_timestamp) CL_API_SUFFIX__VERSION_2_1 {
    GET_SYMBOL_TIMER(clGetDeviceAndHostTimer);
    return function_ptr(device, device_timestamp, host_timestamp);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clGetHostTimer(cl_device_id device,
               cl_ulong *   host_timestamp) CL_API_SUFFIX__VERSION_2_1 {
    GET_SYMBOL_TIMER(clGetHostTimer);
    return function_ptr(device, host_timestamp);
}

#endif

/* Context APIs */
extern CL_API_ENTRY cl_context CL_API_CALL
clCreateContext(const cl_context_properties * properties,
                cl_uint              num_devices,
                const cl_device_id * devices,
                void (CL_CALLBACK * pfn_notify)(const char * errinfo,
                                                const void * private_info,
                                                size_t       cb,
                                                void *       user_data),
                void *               user_data,
                cl_int *             errcode_ret) CL_API_SUFFIX__VERSION_1_0 {
    GET_SYMBOL_TIMER(clCreateContext);
    return function_ptr(properties, num_devices, devices, pfn_notify, user_data, errcode_ret);
}

extern CL_API_ENTRY cl_context CL_API_CALL
clCreateContextFromType(const cl_context_properties * properties,
                        cl_device_type      device_type,
                        void (CL_CALLBACK * pfn_notify)(const char * errinfo,
                                                        const void * private_info,
                                                        size_t       cb,
                                                        void *       user_data),
                        void *              user_data,
                        cl_int *            errcode_ret) CL_API_SUFFIX__VERSION_1_0 {
    GET_SYMBOL_TIMER(clCreateContextFromType);
    return function_ptr(properties, device_type, pfn_notify, user_data, errcode_ret);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clRetainContext(cl_context context) CL_API_SUFFIX__VERSION_1_0 {
    GET_SYMBOL_TIMER(clRetainContext);
    return function_ptr(context);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clReleaseContext(cl_context context) CL_API_SUFFIX__VERSION_1_0 {
    GET_SYMBOL_TIMER(clReleaseContext);
    return function_ptr(context);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clGetContextInfo(cl_context         context,
                 cl_context_info    param_name,
                 size_t             param_value_size,
                 void *             param_value,
                 size_t *           param_value_size_ret) CL_API_SUFFIX__VERSION_1_0 {
    GET_SYMBOL_TIMER(clGetContextInfo);
    return function_ptr(context, param_name, param_value_size, param_value, param_value_size_ret);
}

#ifdef CL_VERSION_3_0

extern CL_API_ENTRY cl_int CL_API_CALL
clSetContextDestructorCallback(cl_context         context,
                               void (CL_CALLBACK* pfn_notify)(cl_context context,
                                                              void* user_data),
                               void*              user_data) CL_API_SUFFIX__VERSION_3_0 {
    GET_SYMBOL_TIMER(clSetContextDestructorCallback);
    return function_ptr(context, pfn_notify, user_data);
}

#endif

/* Command Queue APIs */

#ifdef CL_VERSION_2_0

extern CL_API_ENTRY cl_command_queue CL_API_CALL
clCreateCommandQueueWithProperties(cl_context               context,
                                   cl_device_id             device,
                                   const cl_queue_properties *    properties,
                                   cl_int *                 errcode_ret) CL_API_SUFFIX__VERSION_2_0 {
    GET_SYMBOL_TIMER(clCreateCommandQueueWithProperties);
    /* todo - add profiling request! */
    return function_ptr(context, device, properties, errcode_ret);
}

#endif

extern CL_API_ENTRY cl_int CL_API_CALL
clRetainCommandQueue(cl_command_queue command_queue) CL_API_SUFFIX__VERSION_1_0 {
    GET_SYMBOL_TIMER(clRetainCommandQueue);
    return function_ptr(command_queue);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clReleaseCommandQueue(cl_command_queue command_queue) CL_API_SUFFIX__VERSION_1_0 {
    GET_SYMBOL_TIMER(clReleaseCommandQueue);
    return function_ptr(command_queue);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clGetCommandQueueInfo(cl_command_queue      command_queue,
                      cl_command_queue_info param_name,
                      size_t                param_value_size,
                      void *                param_value,
                      size_t *              param_value_size_ret) CL_API_SUFFIX__VERSION_1_0 {
    GET_SYMBOL_TIMER(clGetCommandQueueInfo);
    return function_ptr(command_queue, param_name, param_value_size, param_value, param_value_size_ret);
}

/* Memory Object APIs */
extern CL_API_ENTRY cl_mem CL_API_CALL
clCreateBuffer(cl_context   context,
               cl_mem_flags flags,
               size_t       size,
               void *       host_ptr,
               cl_int *     errcode_ret) CL_API_SUFFIX__VERSION_1_0 {
    GET_SYMBOL_TIMER(clCreateBuffer);
    return function_ptr(context, flags, size, host_ptr, errcode_ret);
}

#ifdef CL_VERSION_1_1

extern CL_API_ENTRY cl_mem CL_API_CALL
clCreateSubBuffer(cl_mem                   buffer,
                  cl_mem_flags             flags,
                  cl_buffer_create_type    buffer_create_type,
                  const void *             buffer_create_info,
                  cl_int *                 errcode_ret) CL_API_SUFFIX__VERSION_1_1 {
    GET_SYMBOL_TIMER(clCreateSubBuffer);
    return function_ptr(buffer, flags, buffer_create_type, buffer_create_info, errcode_ret);
}

#endif

#ifdef CL_VERSION_1_2

extern CL_API_ENTRY cl_mem CL_API_CALL
clCreateImage(cl_context              context,
              cl_mem_flags            flags,
              const cl_image_format * image_format,
              const cl_image_desc *   image_desc,
              void *                  host_ptr,
              cl_int *                errcode_ret) CL_API_SUFFIX__VERSION_1_2 {
    GET_SYMBOL_TIMER(clCreateImage);
    return function_ptr(context, flags, image_format, image_desc, host_ptr, errcode_ret);
}

#endif

#ifdef CL_VERSION_2_0

extern CL_API_ENTRY cl_mem CL_API_CALL
clCreatePipe(cl_context                 context,
             cl_mem_flags               flags,
             cl_uint                    pipe_packet_size,
             cl_uint                    pipe_max_packets,
             const cl_pipe_properties * properties,
             cl_int *                   errcode_ret) CL_API_SUFFIX__VERSION_2_0 {
    GET_SYMBOL_TIMER(clCreatePipe);
    return function_ptr(context, flags, pipe_packet_size, pipe_max_packets, properties, errcode_ret);
}

#endif

#ifdef CL_VERSION_3_0

extern CL_API_ENTRY cl_mem CL_API_CALL
clCreateBufferWithProperties(cl_context                context,
                             const cl_mem_properties * properties,
                             cl_mem_flags              flags,
                             size_t                    size,
                             void *                    host_ptr,
                             cl_int *                  errcode_ret) CL_API_SUFFIX__VERSION_3_0 {
    GET_SYMBOL_TIMER(clCreateBufferWithProperties);
    return function_ptr(context, properties, flags, size, host_prt, errcode_ret);
}

extern CL_API_ENTRY cl_mem CL_API_CALL
clCreateImageWithProperties(cl_context                context,
                            const cl_mem_properties * properties,
                            cl_mem_flags              flags,
                            const cl_image_format *   image_format,
                            const cl_image_desc *     image_desc,
                            void *                    host_ptr,
                            cl_int *                  errcode_ret) CL_API_SUFFIX__VERSION_3_0 {
    GET_SYMBOL_TIMER(clCreateImageWithProperties);
    return function_ptr(context, properties, flags, image_format, image_desc, host_ptr, errcode_ret);
}

#endif

extern CL_API_ENTRY cl_int CL_API_CALL
clRetainMemObject(cl_mem memobj) CL_API_SUFFIX__VERSION_1_0 {
    GET_SYMBOL_TIMER(clRetainMemObject);
    return function_ptr(memobj);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clReleaseMemObject(cl_mem memobj) CL_API_SUFFIX__VERSION_1_0 {
    GET_SYMBOL_TIMER(clReleaseMemObject);
    return function_ptr(memobj);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clGetSupportedImageFormats(cl_context           context,
                           cl_mem_flags         flags,
                           cl_mem_object_type   image_type,
                           cl_uint              num_entries,
                           cl_image_format *    image_formats,
                           cl_uint *            num_image_formats) CL_API_SUFFIX__VERSION_1_0 {
    GET_SYMBOL_TIMER(clGetSupportedImageFormats);
    return function_ptr(context, flags, image_type, num_entries, image_formats, num_image_formats);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clGetMemObjectInfo(cl_mem           memobj,
                   cl_mem_info      param_name,
                   size_t           param_value_size,
                   void *           param_value,
                   size_t *         param_value_size_ret) CL_API_SUFFIX__VERSION_1_0 {
    GET_SYMBOL_TIMER(clGetMemObjectInfo);
    return function_ptr(memobj, param_name, param_value_size, param_value, param_value_size_ret);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clGetImageInfo(cl_mem           image,
               cl_image_info    param_name,
               size_t           param_value_size,
               void *           param_value,
               size_t *         param_value_size_ret) CL_API_SUFFIX__VERSION_1_0 {
    GET_SYMBOL_TIMER(clGetImageInfo);
    return function_ptr(image, param_name, param_value_size, param_value, param_value_size_ret);
}

#ifdef CL_VERSION_2_0

extern CL_API_ENTRY cl_int CL_API_CALL
clGetPipeInfo(cl_mem           pipe,
              cl_pipe_info     param_name,
              size_t           param_value_size,
              void *           param_value,
              size_t *         param_value_size_ret) CL_API_SUFFIX__VERSION_2_0 {
    GET_SYMBOL_TIMER(clGetPipeInfo);
    return function_ptr(pipe, param_name, param_value_size, param_value, param_value_size_ret);
}

#endif

#ifdef CL_VERSION_1_1

extern CL_API_ENTRY cl_int CL_API_CALL
clSetMemObjectDestructorCallback(cl_mem memobj,
                                 void (CL_CALLBACK * pfn_notify)(cl_mem memobj,
                                                                 void * user_data),
                                 void * user_data) CL_API_SUFFIX__VERSION_1_1 {
    GET_SYMBOL_TIMER(clSetMemObjectDestructorCallback);
    return function_ptr(memobj, pfn_notify, user_data);
}

#endif

/* SVM Allocation APIs */

#ifdef CL_VERSION_2_0

extern CL_API_ENTRY void * CL_API_CALL
clSVMAlloc(cl_context       context,
           cl_svm_mem_flags flags,
           size_t           size,
           cl_uint          alignment) CL_API_SUFFIX__VERSION_2_0 {
    GET_SYMBOL_TIMER(clSVMAlloc);
    return function_ptr(context, flags, size, alignment);
}

extern CL_API_ENTRY void CL_API_CALL
clSVMFree(cl_context        context,
          void *            svm_pointer) CL_API_SUFFIX__VERSION_2_0 {
    GET_SYMBOL_TIMER(clSVMFree);
    return function_ptr(context, svm_pointer);
}

#endif

/* Sampler APIs */

#ifdef CL_VERSION_2_0

extern CL_API_ENTRY cl_sampler CL_API_CALL
clCreateSamplerWithProperties(cl_context                     context,
                              const cl_sampler_properties *  sampler_properties,
                              cl_int *                       errcode_ret) CL_API_SUFFIX__VERSION_2_0 {
    GET_SYMBOL_TIMER(clCreateSamplerWithProperties);
    return function_ptr(context, sampler_properties, errcode_ret);
}

#endif

extern CL_API_ENTRY cl_int CL_API_CALL
clRetainSampler(cl_sampler sampler) CL_API_SUFFIX__VERSION_1_0 {
    GET_SYMBOL_TIMER(clRetainSampler);
    return function_ptr(sampler);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clReleaseSampler(cl_sampler sampler) CL_API_SUFFIX__VERSION_1_0 {
    GET_SYMBOL_TIMER(clReleaseSampler);
    return function_ptr(sampler);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clGetSamplerInfo(cl_sampler         sampler,
                 cl_sampler_info    param_name,
                 size_t             param_value_size,
                 void *             param_value,
                 size_t *           param_value_size_ret) CL_API_SUFFIX__VERSION_1_0 {
    GET_SYMBOL_TIMER(clGetSamplerInfo);
    return function_ptr(sampler, param_name, param_value_size, param_value, param_value_size_ret);
}
////////////////

/* Program Object APIs */
extern CL_API_ENTRY cl_program CL_API_CALL
clCreateProgramWithSource(cl_context        context,
                          cl_uint           count,
                          const char **     strings,
                          const size_t *    lengths,
                          cl_int *          errcode_ret) CL_API_SUFFIX__VERSION_1_0 {
    GET_SYMBOL_TIMER(clCreateProgramWithSource);
    return function_ptr(context, count, strings, lengths, errcode_ret);
}

extern CL_API_ENTRY cl_program CL_API_CALL
clCreateProgramWithBinary(cl_context                     context,
                          cl_uint                        num_devices,
                          const cl_device_id *           device_list,
                          const size_t *                 lengths,
                          const unsigned char **         binaries,
                          cl_int *                       binary_status,
                          cl_int *                       errcode_ret) CL_API_SUFFIX__VERSION_1_0 {
    GET_SYMBOL_TIMER(clCreateProgramWithBinary);
    return function_ptr(context, num_devices, device_list, lengths, binaries, binary_status, errcode_ret);
}

#ifdef CL_VERSION_1_2

extern CL_API_ENTRY cl_program CL_API_CALL
clCreateProgramWithBuiltInKernels(cl_context            context,
                                  cl_uint               num_devices,
                                  const cl_device_id *  device_list,
                                  const char *          kernel_names,
                                  cl_int *              errcode_ret) CL_API_SUFFIX__VERSION_1_2 {
    GET_SYMBOL_TIMER(clCreateProgramWithBuiltInKernels);
    return function_ptr(context, num_devices, device_list, kernel_names, errcode_ret);
}

#endif

#ifdef CL_VERSION_2_1

extern CL_API_ENTRY cl_program CL_API_CALL
clCreateProgramWithIL(cl_context    context,
                     const void*    il,
                     size_t         length,
                     cl_int*        errcode_ret) CL_API_SUFFIX__VERSION_2_1 {
    GET_SYMBOL_TIMER(clCreateProgramWithIL);
    return function_ptr(context, il, length, errcode_ret);
}

#endif

extern CL_API_ENTRY cl_int CL_API_CALL
clRetainProgram(cl_program program) CL_API_SUFFIX__VERSION_1_0 {
    GET_SYMBOL_TIMER(clRetainProgram);
    return function_ptr(program);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clReleaseProgram(cl_program program) CL_API_SUFFIX__VERSION_1_0 {
    GET_SYMBOL_TIMER(clReleaseProgram);
    return function_ptr(program);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clBuildProgram(cl_program           program,
               cl_uint              num_devices,
               const cl_device_id * device_list,
               const char *         options,
               void (CL_CALLBACK *  pfn_notify)(cl_program program,
                                                void * user_data),
               void *               user_data) CL_API_SUFFIX__VERSION_1_0 {
    GET_SYMBOL_TIMER(clBuildProgram);
    return function_ptr(program, num_devices, device_list, options, pfn_notify, user_data);
}

#ifdef CL_VERSION_1_2

extern CL_API_ENTRY cl_int CL_API_CALL
clCompileProgram(cl_program           program,
                 cl_uint              num_devices,
                 const cl_device_id * device_list,
                 const char *         options,
                 cl_uint              num_input_headers,
                 const cl_program *   input_headers,
                 const char **        header_include_names,
                 void (CL_CALLBACK *  pfn_notify)(cl_program program,
                                                  void * user_data),
                 void *               user_data) CL_API_SUFFIX__VERSION_1_2 {
    GET_SYMBOL_TIMER(clCompileProgram);
    return function_ptr(program, num_devices, device_list, options, num_input_headers, input_headers, header_include_names, pfn_notify, user_data);
}

extern CL_API_ENTRY cl_program CL_API_CALL
clLinkProgram(cl_context           context,
              cl_uint              num_devices,
              const cl_device_id * device_list,
              const char *         options,
              cl_uint              num_input_programs,
              const cl_program *   input_programs,
              void (CL_CALLBACK *  pfn_notify)(cl_program program,
                                               void * user_data),
              void *               user_data,
              cl_int *             errcode_ret) CL_API_SUFFIX__VERSION_1_2 {
    GET_SYMBOL_TIMER(clLinkProgram);
    return function_ptr(context, num_devices, device_list, options, num_input_programs, input_programs, pfn_notify, user_data, errcode_ret);
}

#endif

#ifdef CL_VERSION_2_2

extern CL_API_ENTRY CL_API_PREFIX__VERSION_2_2_DEPRECATED cl_int CL_API_CALL
clSetProgramReleaseCallback(cl_program          program,
                            void (CL_CALLBACK * pfn_notify)(cl_program program,
                                                            void * user_data),
                            void *              user_data) CL_API_SUFFIX__VERSION_2_2_DEPRECATED {
    GET_SYMBOL_TIMER(clSetProgramReleaseCallback);
    return function_ptr(program, pfn_notify, user_data);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clSetProgramSpecializationConstant(cl_program  program,
                                   cl_uint     spec_id,
                                   size_t      spec_size,
                                   const void* spec_value) CL_API_SUFFIX__VERSION_2_2 {
    GET_SYMBOL_TIMER(clSetProgramSpecializationConstant);
    return function_ptr(program, spec_id, spec_size, spec_value);
}

#endif

#ifdef CL_VERSION_1_2

extern CL_API_ENTRY cl_int CL_API_CALL
clUnloadPlatformCompiler(cl_platform_id platform) CL_API_SUFFIX__VERSION_1_2 {
    GET_SYMBOL_TIMER(clUnloadPlatformCompiler);
    return function_ptr(platform);
}

#endif

extern CL_API_ENTRY cl_int CL_API_CALL
clGetProgramInfo(cl_program         program,
                 cl_program_info    param_name,
                 size_t             param_value_size,
                 void *             param_value,
                 size_t *           param_value_size_ret) CL_API_SUFFIX__VERSION_1_0 {
    GET_SYMBOL_TIMER(clGetProgramInfo);
    return function_ptr(program, param_name, param_value_size, param_value, param_value_size_ret);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clGetProgramBuildInfo(cl_program            program,
                      cl_device_id          device,
                      cl_program_build_info param_name,
                      size_t                param_value_size,
                      void *                param_value,
                      size_t *              param_value_size_ret) CL_API_SUFFIX__VERSION_1_0 {
    GET_SYMBOL_TIMER(clGetProgramBuildInfo);
    return function_ptr(program, device, param_name, param_value_size, param_value, param_value_size_ret);
}

/////////////
/* Kernel Object APIs */
extern CL_API_ENTRY cl_kernel CL_API_CALL
clCreateKernel(cl_program      program,
               const char *    kernel_name,
               cl_int *        errcode_ret) CL_API_SUFFIX__VERSION_1_0 {
    GET_SYMBOL_TIMER(clCreateKernel);
    return function_ptr(program, kernel_name, errcode_ret);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clCreateKernelsInProgram(cl_program     program,
                         cl_uint        num_kernels,
                         cl_kernel *    kernels,
                         cl_uint *      num_kernels_ret) CL_API_SUFFIX__VERSION_1_0 {
    GET_SYMBOL_TIMER(clCreateKernelsInProgram);
    return function_ptr(program, num_kernels, kernels, num_kernels_ret);
}

#ifdef CL_VERSION_2_1

extern CL_API_ENTRY cl_kernel CL_API_CALL
clCloneKernel(cl_kernel     source_kernel,
              cl_int*       errcode_ret) CL_API_SUFFIX__VERSION_2_1 {
    GET_SYMBOL_TIMER(clCloneKernel);
    return function_ptr(source_kernel, errcode_ret);
}

#endif

extern CL_API_ENTRY cl_int CL_API_CALL
clRetainKernel(cl_kernel    kernel) CL_API_SUFFIX__VERSION_1_0 {
    GET_SYMBOL_TIMER(clRetainKernel);
    return function_ptr(kernel);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clReleaseKernel(cl_kernel   kernel) CL_API_SUFFIX__VERSION_1_0 {
    GET_SYMBOL_TIMER(clReleaseKernel);
    return function_ptr(kernel);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clSetKernelArg(cl_kernel    kernel,
               cl_uint      arg_index,
               size_t       arg_size,
               const void * arg_value) CL_API_SUFFIX__VERSION_1_0 {
    GET_SYMBOL_TIMER(clSetKernelArg);
    return function_ptr(kernel, arg_index, arg_size, arg_value);
}

#ifdef CL_VERSION_2_0

extern CL_API_ENTRY cl_int CL_API_CALL
clSetKernelArgSVMPointer(cl_kernel    kernel,
                         cl_uint      arg_index,
                         const void * arg_value) CL_API_SUFFIX__VERSION_2_0 {
    GET_SYMBOL_TIMER(clSetKernelArgSVMPointer);
    return function_ptr(kernel, arg_index, arg_value);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clSetKernelExecInfo(cl_kernel            kernel,
                    cl_kernel_exec_info  param_name,
                    size_t               param_value_size,
                    const void *         param_value) CL_API_SUFFIX__VERSION_2_0 {
    GET_SYMBOL_TIMER(clSetKernelExecInfo);
    return function_ptr(kernel, param_name, param_value_size, param_value);
}

#endif

extern CL_API_ENTRY cl_int CL_API_CALL
clGetKernelInfo(cl_kernel       kernel,
                cl_kernel_info  param_name,
                size_t          param_value_size,
                void *          param_value,
                size_t *        param_value_size_ret) CL_API_SUFFIX__VERSION_1_0 {
    GET_SYMBOL_TIMER(clGetKernelInfo);
    return function_ptr(kernel, param_name, param_value_size, param_value, param_value_size_ret);
}

#ifdef CL_VERSION_1_2

extern CL_API_ENTRY cl_int CL_API_CALL
clGetKernelArgInfo(cl_kernel       kernel,
                   cl_uint         arg_index,
                   cl_kernel_arg_info  param_name,
                   size_t          param_value_size,
                   void *          param_value,
                   size_t *        param_value_size_ret) CL_API_SUFFIX__VERSION_1_2 {
    GET_SYMBOL_TIMER(clGetKernelArgInfo);
    return function_ptr(kernel, arg_index, param_name, param_value_size, param_value, param_value_size_ret);
}

#endif

extern CL_API_ENTRY cl_int CL_API_CALL
clGetKernelWorkGroupInfo(cl_kernel                  kernel,
                         cl_device_id               device,
                         cl_kernel_work_group_info  param_name,
                         size_t                     param_value_size,
                         void *                     param_value,
                         size_t *                   param_value_size_ret) CL_API_SUFFIX__VERSION_1_0 {
    GET_SYMBOL_TIMER(clGetKernelWorkGroupInfo);
    return function_ptr(kernel, device, param_name, param_value_size, param_value, param_value_size_ret);
}

#ifdef CL_VERSION_2_1

extern CL_API_ENTRY cl_int CL_API_CALL
clGetKernelSubGroupInfo(cl_kernel                   kernel,
                        cl_device_id                device,
                        cl_kernel_sub_group_info    param_name,
                        size_t                      input_value_size,
                        const void*                 input_value,
                        size_t                      param_value_size,
                        void*                       param_value,
                        size_t*                     param_value_size_ret) CL_API_SUFFIX__VERSION_2_1 {
    GET_SYMBOL_TIMER(clGetKernelSubGroupInfo);
    return function_ptr(kernel, device, param_name, input_value_size, input_value, param_value_size, param_value, param_value_size_ret);
}

#endif

////////////////
/* Event Object APIs */
extern CL_API_ENTRY cl_int CL_API_CALL
clWaitForEvents(cl_uint             num_events,
                const cl_event *    event_list) CL_API_SUFFIX__VERSION_1_0 {
    GET_SYMBOL_TIMER(clWaitForEvents);
    return function_ptr(num_events, event_list);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clGetEventInfo(cl_event         event,
               cl_event_info    param_name,
               size_t           param_value_size,
               void *           param_value,
               size_t *         param_value_size_ret) CL_API_SUFFIX__VERSION_1_0 {
    GET_SYMBOL_TIMER(clGetEventInfo);
    return function_ptr(event, param_name, param_value_size, param_value, param_value_size_ret);
}

#ifdef CL_VERSION_1_1

extern CL_API_ENTRY cl_event CL_API_CALL
clCreateUserEvent(cl_context    context,
                  cl_int *      errcode_ret) CL_API_SUFFIX__VERSION_1_1 {
    GET_SYMBOL_TIMER(clCreateUserEvent);
    return function_ptr(context, errcode_ret);
}

#endif

extern CL_API_ENTRY cl_int CL_API_CALL
clRetainEvent(cl_event event) CL_API_SUFFIX__VERSION_1_0 {
    GET_SYMBOL_TIMER(clRetainEvent);
    return function_ptr(event);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clReleaseEvent(cl_event event) CL_API_SUFFIX__VERSION_1_0 {
    GET_SYMBOL_TIMER(clReleaseEvent);
    return function_ptr(event);
}

#ifdef CL_VERSION_1_1

extern CL_API_ENTRY cl_int CL_API_CALL
clSetUserEventStatus(cl_event   event,
                     cl_int     execution_status) CL_API_SUFFIX__VERSION_1_1 {
    GET_SYMBOL_TIMER(clSetUserEventStatus);
    return function_ptr(event, execution_status);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clSetEventCallback(cl_event    event,
                   cl_int      command_exec_callback_type,
                   void (CL_CALLBACK * pfn_notify)(cl_event event,
                                                   cl_int   event_command_status,
                                                   void *   user_data),
                   void *      user_data) CL_API_SUFFIX__VERSION_1_1 {
    GET_SYMBOL_TIMER(clSetEventCallback);
    return function_ptr(event, command_exec_callback_type, pfn_notify, user_data);
}

#endif

#if 0
/* Profiling APIs */
extern CL_API_ENTRY cl_int CL_API_CALL
clGetEventProfilingInfo(cl_event            event,
                        cl_profiling_info   param_name,
                        size_t              param_value_size,
                        void *              param_value,
                        size_t *            param_value_size_ret) CL_API_SUFFIX__VERSION_1_0;
#endif

//////////
/* Flush and Finish APIs */
extern CL_API_ENTRY cl_int CL_API_CALL
clFlush(cl_command_queue command_queue) CL_API_SUFFIX__VERSION_1_0 {
    GET_SYMBOL_TIMER(clFlush);
    return function_ptr(command_queue);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clFinish(cl_command_queue command_queue) CL_API_SUFFIX__VERSION_1_0 {
    GET_SYMBOL_TIMER(clFinish);
    return function_ptr(command_queue);
}

/////////
/* Enqueued Commands APIs */
extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueReadBuffer(cl_command_queue    command_queue,
                    cl_mem              buffer,
                    cl_bool             blocking_read,
                    size_t              offset,
                    size_t              size,
                    void *              ptr,
                    cl_uint             num_events_in_wait_list,
                    const cl_event *    event_wait_list,
                    cl_event *          event) CL_API_SUFFIX__VERSION_1_0 {
    GET_SYMBOL_TIMER(clEnqueueReadBuffer);
    return function_ptr(command_queue, buffer, blocking_read, offset, size, ptr, num_events_in_wait_list, event_wait_list, event);
}

#ifdef CL_VERSION_1_1

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueReadBufferRect(cl_command_queue    command_queue,
                        cl_mem              buffer,
                        cl_bool             blocking_read,
                        const size_t *      buffer_origin,
                        const size_t *      host_origin,
                        const size_t *      region,
                        size_t              buffer_row_pitch,
                        size_t              buffer_slice_pitch,
                        size_t              host_row_pitch,
                        size_t              host_slice_pitch,
                        void *              ptr,
                        cl_uint             num_events_in_wait_list,
                        const cl_event *    event_wait_list,
                        cl_event *          event) CL_API_SUFFIX__VERSION_1_1 {
    GET_SYMBOL_TIMER(clEnqueueReadBufferRect);
    return function_ptr(command_queue, buffer, blocking_read, buffer_origin, host_origin, region, buffer_row_pitch, buffer_slice_pitch, host_row_pitch, host_slice_pitch, ptr, num_events_in_wait_list, event_wait_list, event);
}

#endif

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueWriteBuffer(cl_command_queue   command_queue,
                     cl_mem             buffer,
                     cl_bool            blocking_write,
                     size_t             offset,
                     size_t             size,
                     const void *       ptr,
                     cl_uint            num_events_in_wait_list,
                     const cl_event *   event_wait_list,
                     cl_event *         event) CL_API_SUFFIX__VERSION_1_0 {
    GET_SYMBOL_TIMER(clEnqueueWriteBuffer);
    return function_ptr(command_queue, buffer, blocking_write, offset, size, ptr, num_events_in_wait_list, event_wait_list, event);
}

#ifdef CL_VERSION_1_1

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueWriteBufferRect(cl_command_queue    command_queue,
                         cl_mem              buffer,
                         cl_bool             blocking_write,
                         const size_t *      buffer_origin,
                         const size_t *      host_origin,
                         const size_t *      region,
                         size_t              buffer_row_pitch,
                         size_t              buffer_slice_pitch,
                         size_t              host_row_pitch,
                         size_t              host_slice_pitch,
                         const void *        ptr,
                         cl_uint             num_events_in_wait_list,
                         const cl_event *    event_wait_list,
                         cl_event *          event) CL_API_SUFFIX__VERSION_1_1 {
    GET_SYMBOL_TIMER(clEnqueueWriteBufferRect);
    return function_ptr(command_queue, buffer, blocking_write, buffer_origin, host_origin, region, buffer_row_pitch, buffer_slice_pitch, host_row_pitch, host_slice_pitch, ptr, num_events_in_wait_list, event_wait_list, event);
}

#endif

#ifdef CL_VERSION_1_2

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueFillBuffer(cl_command_queue   command_queue,
                    cl_mem             buffer,
                    const void *       pattern,
                    size_t             pattern_size,
                    size_t             offset,
                    size_t             size,
                    cl_uint            num_events_in_wait_list,
                    const cl_event *   event_wait_list,
                    cl_event *         event) CL_API_SUFFIX__VERSION_1_2 {
    GET_SYMBOL_TIMER(clEnqueueFillBuffer);
    return function_ptr(command_queue, buffer, pattern, pattern_size, offset, size, num_events_in_wait_list, event_wait_list, event);
}

#endif

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueCopyBuffer(cl_command_queue    command_queue,
                    cl_mem              src_buffer,
                    cl_mem              dst_buffer,
                    size_t              src_offset,
                    size_t              dst_offset,
                    size_t              size,
                    cl_uint             num_events_in_wait_list,
                    const cl_event *    event_wait_list,
                    cl_event *          event) CL_API_SUFFIX__VERSION_1_0 {
    GET_SYMBOL_TIMER(clEnqueueCopyBuffer);
    return function_ptr(command_queue, src_buffer, dst_buffer, src_offset, dst_offset, size, num_events_in_wait_list, event_wait_list, event);
}

#ifdef CL_VERSION_1_1

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueCopyBufferRect(cl_command_queue    command_queue,
                        cl_mem              src_buffer,
                        cl_mem              dst_buffer,
                        const size_t *      src_origin,
                        const size_t *      dst_origin,
                        const size_t *      region,
                        size_t              src_row_pitch,
                        size_t              src_slice_pitch,
                        size_t              dst_row_pitch,
                        size_t              dst_slice_pitch,
                        cl_uint             num_events_in_wait_list,
                        const cl_event *    event_wait_list,
                        cl_event *          event) CL_API_SUFFIX__VERSION_1_1 {
    GET_SYMBOL_TIMER(clEnqueueCopyBufferRect);
    return function_ptr(command_queue, src_buffer, dst_buffer, src_origin, dst_origin, region, src_row_pitch, src_slice_pitch, dst_row_pitch, dst_slice_pitch, num_events_in_wait_list, event_wait_list, event);
}

#endif

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueReadImage(cl_command_queue     command_queue,
                   cl_mem               image,
                   cl_bool              blocking_read,
                   const size_t *       origin,
                   const size_t *       region,
                   size_t               row_pitch,
                   size_t               slice_pitch,
                   void *               ptr,
                   cl_uint              num_events_in_wait_list,
                   const cl_event *     event_wait_list,
                   cl_event *           event) CL_API_SUFFIX__VERSION_1_0 {
    GET_SYMBOL_TIMER(clEnqueueReadImage);
    return function_ptr(command_queue, image, blocking_read, origin, region, row_pitch, slice_pitch, ptr, num_events_in_wait_list, event_wait_list, event);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueWriteImage(cl_command_queue    command_queue,
                    cl_mem              image,
                    cl_bool             blocking_write,
                    const size_t *      origin,
                    const size_t *      region,
                    size_t              input_row_pitch,
                    size_t              input_slice_pitch,
                    const void *        ptr,
                    cl_uint             num_events_in_wait_list,
                    const cl_event *    event_wait_list,
                    cl_event *          event) CL_API_SUFFIX__VERSION_1_0 {
    GET_SYMBOL_TIMER(clEnqueueWriteImage);
    return function_ptr(command_queue, image, blocking_write, origin, region, input_row_pitch, input_slice_pitch, ptr, num_events_in_wait_list, event_wait_list, event);
}

#ifdef CL_VERSION_1_2

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueFillImage(cl_command_queue   command_queue,
                   cl_mem             image,
                   const void *       fill_color,
                   const size_t *     origin,
                   const size_t *     region,
                   cl_uint            num_events_in_wait_list,
                   const cl_event *   event_wait_list,
                   cl_event *         event) CL_API_SUFFIX__VERSION_1_2 {
    GET_SYMBOL_TIMER(clEnqueueFillImage);
    return function_ptr(command_queue, image, fill_color, origin, region, num_events_in_wait_list, event_wait_list, event);
}

#endif

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueCopyImage(cl_command_queue     command_queue,
                   cl_mem               src_image,
                   cl_mem               dst_image,
                   const size_t *       src_origin,
                   const size_t *       dst_origin,
                   const size_t *       region,
                   cl_uint              num_events_in_wait_list,
                   const cl_event *     event_wait_list,
                   cl_event *           event) CL_API_SUFFIX__VERSION_1_0 {
    GET_SYMBOL_TIMER(clEnqueueCopyImage);
    return function_ptr(command_queue, src_image, dst_image, src_origin, dst_origin, region, num_events_in_wait_list, event_wait_list, event);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueCopyImageToBuffer(cl_command_queue command_queue,
                           cl_mem           src_image,
                           cl_mem           dst_buffer,
                           const size_t *   src_origin,
                           const size_t *   region,
                           size_t           dst_offset,
                           cl_uint          num_events_in_wait_list,
                           const cl_event * event_wait_list,
                           cl_event *       event) CL_API_SUFFIX__VERSION_1_0 {
    GET_SYMBOL_TIMER(clEnqueueCopyImageToBuffer);
    return function_ptr(command_queue, src_image, dst_buffer, src_origin, region, dst_offset, num_events_in_wait_list, event_wait_list, event);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueCopyBufferToImage(cl_command_queue command_queue,
                           cl_mem           src_buffer,
                           cl_mem           dst_image,
                           size_t           src_offset,
                           const size_t *   dst_origin,
                           const size_t *   region,
                           cl_uint          num_events_in_wait_list,
                           const cl_event * event_wait_list,
                           cl_event *       event) CL_API_SUFFIX__VERSION_1_0 {
    GET_SYMBOL_TIMER(clEnqueueCopyBufferToImage);
    return function_ptr(command_queue, src_buffer, dst_image, src_offset, dst_origin, region, num_events_in_wait_list, event_wait_list, event);
}

extern CL_API_ENTRY void * CL_API_CALL
clEnqueueMapBuffer(cl_command_queue command_queue,
                   cl_mem           buffer,
                   cl_bool          blocking_map,
                   cl_map_flags     map_flags,
                   size_t           offset,
                   size_t           size,
                   cl_uint          num_events_in_wait_list,
                   const cl_event * event_wait_list,
                   cl_event *       event,
                   cl_int *         errcode_ret) CL_API_SUFFIX__VERSION_1_0 {
    GET_SYMBOL_TIMER(clEnqueueMapBuffer);
    return function_ptr(command_queue, buffer, blocking_map, map_flags, offset, size, num_events_in_wait_list, event_wait_list, event, errcode_ret);
}

extern CL_API_ENTRY void * CL_API_CALL
clEnqueueMapImage(cl_command_queue  command_queue,
                  cl_mem            image,
                  cl_bool           blocking_map,
                  cl_map_flags      map_flags,
                  const size_t *    origin,
                  const size_t *    region,
                  size_t *          image_row_pitch,
                  size_t *          image_slice_pitch,
                  cl_uint           num_events_in_wait_list,
                  const cl_event *  event_wait_list,
                  cl_event *        event,
                  cl_int *          errcode_ret) CL_API_SUFFIX__VERSION_1_0 {
    GET_SYMBOL_TIMER(clEnqueueMapImage);
    return function_ptr(command_queue, image, blocking_map, map_flags, origin, region, image_row_pitch, image_slice_pitch, num_events_in_wait_list, event_wait_list, event, errcode_ret);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueUnmapMemObject(cl_command_queue command_queue,
                        cl_mem           memobj,
                        void *           mapped_ptr,
                        cl_uint          num_events_in_wait_list,
                        const cl_event * event_wait_list,
                        cl_event *       event) CL_API_SUFFIX__VERSION_1_0 {
    GET_SYMBOL_TIMER(clEnqueueUnmapMemObject);
    return function_ptr(command_queue, memobj, mapped_ptr, num_events_in_wait_list, event_wait_list, event);
}

#ifdef CL_VERSION_1_2

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueMigrateMemObjects(cl_command_queue       command_queue,
                           cl_uint                num_mem_objects,
                           const cl_mem *         mem_objects,
                           cl_mem_migration_flags flags,
                           cl_uint                num_events_in_wait_list,
                           const cl_event *       event_wait_list,
                           cl_event *             event) CL_API_SUFFIX__VERSION_1_2 {
    GET_SYMBOL_TIMER(clEnqueueMigrateMemObjects);
    return function_ptr(command_queue, num_mem_objects, mem_objects, flags, num_events_in_wait_list, event_wait_list, event);
}

#endif

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueNDRangeKernel(cl_command_queue command_queue,
                       cl_kernel        kernel,
                       cl_uint          work_dim,
                       const size_t *   global_work_offset,
                       const size_t *   global_work_size,
                       const size_t *   local_work_size,
                       cl_uint          num_events_in_wait_list,
                       const cl_event * event_wait_list,
                       cl_event *       event) CL_API_SUFFIX__VERSION_1_0 {
    GET_SYMBOL_TIMER(clEnqueueNDRangeKernel);
    return function_ptr(command_queue, kernel, work_dim, global_work_offset, global_work_size, local_work_size, num_events_in_wait_list, event_wait_list, event);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueNativeKernel(cl_command_queue  command_queue,
                      void (CL_CALLBACK * user_func)(void *),
                      void *            args,
                      size_t            cb_args,
                      cl_uint           num_mem_objects,
                      const cl_mem *    mem_list,
                      const void **     args_mem_loc,
                      cl_uint           num_events_in_wait_list,
                      const cl_event *  event_wait_list,
                      cl_event *        event) CL_API_SUFFIX__VERSION_1_0 {
    GET_SYMBOL_TIMER(clEnqueueNativeKernel);
    return function_ptr(command_queue, user_func, args, cb_args, num_mem_objects, mem_list, args_mem_loc, num_events_in_wait_list, event_wait_list, event);
}

#ifdef CL_VERSION_1_2

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueMarkerWithWaitList(cl_command_queue  command_queue,
                            cl_uint           num_events_in_wait_list,
                            const cl_event *  event_wait_list,
                            cl_event *        event) CL_API_SUFFIX__VERSION_1_2 {
    GET_SYMBOL_TIMER(clEnqueueMarkerWithWaitList);
    return function_ptr(command_queue, num_events_in_wait_list, event_wait_list, event);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueBarrierWithWaitList(cl_command_queue  command_queue,
                             cl_uint           num_events_in_wait_list,
                             const cl_event *  event_wait_list,
                             cl_event *        event) CL_API_SUFFIX__VERSION_1_2 {
    GET_SYMBOL_TIMER(clEnqueueBarrierWithWaitList);
    return function_ptr(command_queue, num_events_in_wait_list, event_wait_list, event);
}

#endif

#ifdef CL_VERSION_2_0

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueSVMFree(cl_command_queue  command_queue,
                 cl_uint           num_svm_pointers,
                 void *            svm_pointers[],
                 void (CL_CALLBACK * pfn_free_func)(cl_command_queue queue,
                                                    cl_uint          num_svm_pointers,
                                                    void *           svm_pointers[],
                                                    void *           user_data),
                 void *            user_data,
                 cl_uint           num_events_in_wait_list,
                 const cl_event *  event_wait_list,
                 cl_event *        event) CL_API_SUFFIX__VERSION_2_0 {
    GET_SYMBOL_TIMER(clEnqueueSVMFree);
    return function_ptr(command_queue, num_svm_pointers, svm_pointers, pfn_free_func, user_data, num_events_in_wait_list, event_wait_list, event);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueSVMMemcpy(cl_command_queue  command_queue,
                   cl_bool           blocking_copy,
                   void *            dst_ptr,
                   const void *      src_ptr,
                   size_t            size,
                   cl_uint           num_events_in_wait_list,
                   const cl_event *  event_wait_list,
                   cl_event *        event) CL_API_SUFFIX__VERSION_2_0 {
    GET_SYMBOL_TIMER(clEnqueueSVMMemcpy);
    return function_ptr(command_queue, blocking_copy, dst_ptr, src_ptr, size, num_events_in_wait_list, event_wait_list, event);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueSVMMemFill(cl_command_queue  command_queue,
                    void *            svm_ptr,
                    const void *      pattern,
                    size_t            pattern_size,
                    size_t            size,
                    cl_uint           num_events_in_wait_list,
                    const cl_event *  event_wait_list,
                    cl_event *        event) CL_API_SUFFIX__VERSION_2_0 {
    GET_SYMBOL_TIMER(clEnqueueSVMMemFill);
    return function_ptr(command_queue, svm_ptr, pattern, pattern_size, size, num_events_in_wait_list, event_wait_list, event);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueSVMMap(cl_command_queue  command_queue,
                cl_bool           blocking_map,
                cl_map_flags      flags,
                void *            svm_ptr,
                size_t            size,
                cl_uint           num_events_in_wait_list,
                const cl_event *  event_wait_list,
                cl_event *        event) CL_API_SUFFIX__VERSION_2_0 {
    GET_SYMBOL_TIMER(clEnqueueSVMMap);
    return function_ptr(command_queue, blocking_map, flags, svm_ptr, size, num_events_in_wait_list, event_wait_list, event);
}

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueSVMUnmap(cl_command_queue  command_queue,
                  void *            svm_ptr,
                  cl_uint           num_events_in_wait_list,
                  const cl_event *  event_wait_list,
                  cl_event *        event) CL_API_SUFFIX__VERSION_2_0 {
    GET_SYMBOL_TIMER(clEnqueueSVMUnmap);
    return function_ptr(command_queue, svm_ptr, num_events_in_wait_list, event_wait_list, event);
}

#endif

#ifdef CL_VERSION_2_1

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueSVMMigrateMem(cl_command_queue         command_queue,
                       cl_uint                  num_svm_pointers,
                       const void **            svm_pointers,
                       const size_t *           sizes,
                       cl_mem_migration_flags   flags,
                       cl_uint                  num_events_in_wait_list,
                       const cl_event *         event_wait_list,
                       cl_event *               event) CL_API_SUFFIX__VERSION_2_1 {
    GET_SYMBOL_TIMER(clEnqueueSVMMigrateMem);
    return function_ptr(command_queue, num_svm_pointers, svm_pointers, sizes, flags, num_events_in_wait_list, event_wait_list, event);
}

#endif

#ifdef CL_VERSION_1_2

/* Extension function access
 *
 * Returns the extension function address for the given function name,
 * or NULL if a valid function can not be found.  The client must
 * check to make sure the address is not NULL, before using or
 * calling the returned function address.
 */
extern CL_API_ENTRY void * CL_API_CALL
clGetExtensionFunctionAddressForPlatform(cl_platform_id platform,
                                         const char *   func_name) CL_API_SUFFIX__VERSION_1_2 {
    GET_SYMBOL_TIMER(clGetExtensionFunctionAddressForPlatform);
    return function_ptr(platform, func_name);
}

#endif

#ifdef CL_USE_DEPRECATED_OPENCL_1_0_APIS
    /*
     *  WARNING:
     *     This API introduces mutable state into the OpenCL implementation. It has been REMOVED
     *  to better facilitate thread safety.  The 1.0 API is not thread safe. It is not tested by the
     *  OpenCL 1.1 conformance test, and consequently may not work or may not work dependably.
     *  It is likely to be non-performant. Use of this API is not advised. Use at your own risk.
     *
     *  Software developers previously relying on this API are instructed to set the command queue
     *  properties when creating the queue, instead.
     */
    extern CL_API_ENTRY cl_int CL_API_CALL
    clSetCommandQueueProperty(cl_command_queue              command_queue,
                              cl_command_queue_properties   properties,
                              cl_bool                       enable,
                              cl_command_queue_properties * old_properties) CL_API_SUFFIX__VERSION_1_0_DEPRECATED {
    GET_SYMBOL_TIMER(clSetCommandQueueProperty);
    return function_ptr(command_queue, properties, enable, old_properties);
}
#endif /* CL_USE_DEPRECATED_OPENCL_1_0_APIS */

/* Deprecated OpenCL 1.1 APIs */
extern CL_API_ENTRY CL_API_PREFIX__VERSION_1_1_DEPRECATED cl_mem CL_API_CALL
clCreateImage2D(cl_context              context,
                cl_mem_flags            flags,
                const cl_image_format * image_format,
                size_t                  image_width,
                size_t                  image_height,
                size_t                  image_row_pitch,
                void *                  host_ptr,
                cl_int *                errcode_ret) CL_API_SUFFIX__VERSION_1_1_DEPRECATED {
    GET_SYMBOL_TIMER(clCreateImage2D);
    return function_ptr(context, flags, image_format, image_width, image_height, image_row_pitch, host_ptr, errcode_ret);
}

extern CL_API_ENTRY CL_API_PREFIX__VERSION_1_1_DEPRECATED cl_mem CL_API_CALL
clCreateImage3D(cl_context              context,
                cl_mem_flags            flags,
                const cl_image_format * image_format,
                size_t                  image_width,
                size_t                  image_height,
                size_t                  image_depth,
                size_t                  image_row_pitch,
                size_t                  image_slice_pitch,
                void *                  host_ptr,
                cl_int *                errcode_ret) CL_API_SUFFIX__VERSION_1_1_DEPRECATED {
    GET_SYMBOL_TIMER(clCreateImage3D);
    return function_ptr(context, flags, image_format, image_width, image_height, image_depth, image_row_pitch, image_slice_pitch, host_ptr, errcode_ret);
}

extern CL_API_ENTRY CL_API_PREFIX__VERSION_1_1_DEPRECATED cl_int CL_API_CALL
clEnqueueMarker(cl_command_queue    command_queue,
                cl_event *          event) CL_API_SUFFIX__VERSION_1_1_DEPRECATED {
    GET_SYMBOL_TIMER(clEnqueueMarker);
    return function_ptr(command_queue, event);
}

extern CL_API_ENTRY CL_API_PREFIX__VERSION_1_1_DEPRECATED cl_int CL_API_CALL
clEnqueueWaitForEvents(cl_command_queue  command_queue,
                        cl_uint          num_events,
                        const cl_event * event_list) CL_API_SUFFIX__VERSION_1_1_DEPRECATED {
    GET_SYMBOL_TIMER(clEnqueueWaitForEvents);
    return function_ptr(command_queue, num_events, event_list);
}

extern CL_API_ENTRY CL_API_PREFIX__VERSION_1_1_DEPRECATED cl_int CL_API_CALL
clEnqueueBarrier(cl_command_queue command_queue) CL_API_SUFFIX__VERSION_1_1_DEPRECATED {
    GET_SYMBOL_TIMER(clEnqueueBarrier);
    return function_ptr(command_queue);
}

extern CL_API_ENTRY CL_API_PREFIX__VERSION_1_1_DEPRECATED cl_int CL_API_CALL
clUnloadCompiler(void) CL_API_SUFFIX__VERSION_1_1_DEPRECATED {
    GET_SYMBOL_TIMER(clUnloadCompiler);
    return function_ptr();
}

extern CL_API_ENTRY CL_API_PREFIX__VERSION_1_1_DEPRECATED void * CL_API_CALL
clGetExtensionFunctionAddress(const char * func_name) CL_API_SUFFIX__VERSION_1_1_DEPRECATED {
    GET_SYMBOL_TIMER(clGetExtensionFunctionAddress);
    return function_ptr(func_name);
}

/* Deprecated OpenCL 2.0 APIs */
extern CL_API_ENTRY CL_API_PREFIX__VERSION_1_2_DEPRECATED cl_command_queue CL_API_CALL
clCreateCommandQueue(cl_context                     context,
                     cl_device_id                   device,
                     cl_command_queue_properties    properties,
                     cl_int *                       errcode_ret) CL_API_SUFFIX__VERSION_1_2_DEPRECATED {
    GET_SYMBOL_TIMER(clCreateCommandQueue);
    return function_ptr(context, device, properties, errcode_ret);
}

extern CL_API_ENTRY CL_API_PREFIX__VERSION_1_2_DEPRECATED cl_sampler CL_API_CALL
clCreateSampler(cl_context          context,
                cl_bool             normalized_coords,
                cl_addressing_mode  addressing_mode,
                cl_filter_mode      filter_mode,
                cl_int *            errcode_ret) CL_API_SUFFIX__VERSION_1_2_DEPRECATED {
    GET_SYMBOL_TIMER(clCreateSampler);
    return function_ptr(context, normalized_coords, addressing_mode, filter_mode, errcode_ret);
}

extern CL_API_ENTRY CL_API_PREFIX__VERSION_1_2_DEPRECATED cl_int CL_API_CALL
clEnqueueTask(cl_command_queue  command_queue,
              cl_kernel         kernel,
              cl_uint           num_events_in_wait_list,
              const cl_event *  event_wait_list,
              cl_event *        event) CL_API_SUFFIX__VERSION_1_2_DEPRECATED {
    GET_SYMBOL_TIMER(clEnqueueTask);
    return function_ptr(command_queue, kernel, num_events_in_wait_list, event_wait_list, event);
}

}

