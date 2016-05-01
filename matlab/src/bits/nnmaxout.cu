// @file nnpooling.cu
// @brief Pooling block
// @author Andrea Vedaldi

/*
Copyright (C) 2014-15 Andrea Vedaldi and Karel Lenc.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "nnmaxout.hpp"
#include "impl/maxout.hpp"

#if ENABLE_GPU
#include "datacu.hpp"
#endif

#include <assert.h>

using namespace vl ;

/* ---------------------------------------------------------------- */
/*                                                nnpooling_forward */
/* ---------------------------------------------------------------- */

Error
vl::nnmaxout_forward(vl::Context& context,
                      vl::Tensor output,
                      vl::Tensor data,
                      int numunit, int numpiece)
{
  Error status = vlSuccess ;
	switch (output.getMemoryType()) {
    default:
      assert(false) ;
      return vl::vlErrorUnknown ;
    
#ifdef ENABLE_GPU
    case vl::GPU:
          status = vl::impl::maxout_forward<GPU,float>
          ((float*)output.getMemory(), (float const*)data.getMemory(),
           data.getHeight(), data.getWidth(), data.getDepth() * data.getSize(),
           numunit, numpiece);


      if (status == vlErrorCuda) {
        context.setError(context.getCudaHelper().catchCudaError("maxout_*_forward")) ;
      }
#endif
   }
  return context.passError(status, "maxout_forward: ") ;
}

/* ---------------------------------------------------------------- */
/*                                               nnpooling_backward */
/* ---------------------------------------------------------------- */

Error
vl::nnmaxout_backward(Context& context,
                       Tensor derData,
                       Tensor data,
                       Tensor derPooled,
                       int numunit, int numpiece)
{
  vl::Error status = vlSuccess ;
	switch (derData.getMemoryType()) {
      default:
      assert(false) ;
      return vl::vlErrorUnknown ;
#if ENABLE_GPU
    case vl::GPU:
          status = vl::impl::maxout_backward<GPU,float>
          ((float*)derData.getMemory(), (float const*)data.getMemory(), (float const*)derPooled.getMemory(),
           derData.getHeight(), derData.getWidth(), derData.getDepth() * derData.getSize(),
           numunit, numpiece) ;

      if (status == vlErrorCuda) {
        context.setError(context.getCudaHelper().catchCudaError("maxout*_backward: ")) ;
      }
#endif
	}
  return context.passError(status, "maxout_backward: ") ;
}
