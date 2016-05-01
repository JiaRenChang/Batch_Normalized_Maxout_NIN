 // @file pooling_gpu.cu
// @brief Pooling block implementation (GPU)
// @author Andrea Vedaldi
// @author Karel Lenc

/*
Copyright (C) 2014-15 Andrea Vedaldi and Karel Lenc.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "maxout.hpp"
#include "../datacu.hpp"
#include <assert.h>
#include <float.h>
#include <sm_20_atomic_functions.h>

/* ---------------------------------------------------------------- */
/*                                              pooling_max_forward */
/* ---------------------------------------------------------------- */

template<typename T> __global__ void
maxout_kernel
(T* pooled,
 const T* data,
 const int pooledWidth,
 const int pooledHeight,
 const int pooledVolume,
 const int numunit,
 const int numpiece)
{
  int pooledIndex = threadIdx.x + blockIdx.x * blockDim.x;
  if (pooledIndex < pooledVolume) {
    int thx = pooledIndex % (pooledWidth*pooledHeight);  // which element in pooled    
    int ut  = (pooledIndex / (pooledWidth*pooledHeight)) % numunit; //which unit
    int ntr = pooledIndex / (pooledWidth*pooledHeight*numunit); // which trial 
    
    T bestValue = data[thx + pooledWidth*pooledHeight*(ut + ntr*numunit*numpiece)];  // GET vlaue in data

        //for (int k = 0; k < numpiece ; ++k) {     
        //   bestValue = max(bestValue, data[thx + pooledWidth*pooledHeight*(ut*numpiece+k)]) ;
        // }
        for (int k = 0; k < numpiece ; ++k) {     
           bestValue = max(bestValue, data[thx + pooledWidth*pooledHeight*(ut + k*numunit + ntr*numunit*numpiece)]) ;
        }

    pooled[pooledIndex] = bestValue ;
    
  }
}

template<> vl::Error
vl::impl::maxout_forward<vl::GPU, float>(float* pooled,
                                              float const* data,
                                              size_t height, size_t width, size_t depth,
                                              size_t numunit, size_t numpiece)
{
  int pooledWidth = width;
  int pooledHeight = height;
  int pooledVolume = pooledWidth * pooledHeight * depth / numpiece ;
  
  maxout_kernel<float>
  <<< divideUpwards(pooledVolume, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
  (pooled, data,
   pooledHeight, pooledWidth, pooledVolume,
   numunit, numpiece);

  cudaError_t status = cudaPeekAtLastError() ;
  return (status == cudaSuccess) ? vl::vlSuccess : vl::vlErrorCuda ;
}


/* ---------------------------------------------------------------- */
/*                                             pooling_max_backward */
/* ---------------------------------------------------------------- */

template<typename T> __global__ void
maxout_backward_kernel
(T* derData,
 const T* data,
 const T* derPooled,
 const int pooledWidth,
 const int pooledHeight,
 const int pooledVolume,
 const int numunit,
 const int numpiece)
{
  int pooledIndex = threadIdx.x + blockIdx.x * blockDim.x;
  if (pooledIndex < pooledVolume) {

    int thx = pooledIndex % (pooledWidth*pooledHeight);  // which element in pooled
    int ut  = (pooledIndex / (pooledWidth*pooledHeight)) % numunit; //which unit
    int ntr = pooledIndex / (pooledWidth*pooledHeight*numunit); // which trial   

    //T bestValue = data[thx + pooledWidth*pooledHeight*(ut*numpiece)];  // GET vlaue in data

	T bestValue = data[thx + pooledWidth*pooledHeight*(ut +  ntr*numunit*numpiece)];  // GET vlaue in data

    int bestindex = 0;    
        for (int k = 0; k < numpiece ; ++k) {
			//T value = data[thx + pooledWidth*pooledHeight*(ut*numpiece+k)];
			T value = data[thx + pooledWidth*pooledHeight*(ut + k*numunit +  ntr*numunit*numpiece)];
              
               if (value > bestValue) {
					bestValue = value ;
					bestindex = k;    
				}
        }


    /*
     This is bad, but required to eliminate a race condition when writing
     to bottom_diff.
     Caffe goes the other way around, but requrires remembering the layer
     output, or the maximal indexes.
     atomicAdd(add, val)
     */
    //int dain = thx + pooledWidth*pooledHeight*(ut*numpiece+bestindex);
    int dain = thx + pooledWidth*pooledHeight*(ut + bestindex*numunit +  ntr*numunit*numpiece);
    atomicAdd(derData + dain, derPooled[pooledIndex]) ;
    //derData[dain] = derPooled[pooledIndex];
  }
}

template<> vl::Error
vl::impl::maxout_backward<vl::GPU, float>(float* derData,
                                               float const* data,
                                               float const* derPooled,
                                               size_t height, size_t width, size_t depth,
                                              size_t numunit, size_t numpiece)
{
  int pooledWidth = width;
  int pooledHeight = height;
  int pooledVolume = pooledWidth * pooledHeight * depth /  numpiece;

  maxout_backward_kernel<float>
  <<< divideUpwards(pooledVolume, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
      (derData, data, derPooled,
       pooledHeight, pooledWidth, pooledVolume,
		numunit, numpiece);

  cudaError_t status = cudaPeekAtLastError() ;
  return (status == cudaSuccess) ? vl::vlSuccess : vl::vlErrorCuda ;
}
